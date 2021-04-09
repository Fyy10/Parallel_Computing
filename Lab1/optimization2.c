#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#define MIN(a, b) ((a) < (b) ? (a): (b))

int main(int argc, char *argv[]) {
    int count;              // local prime count
    double elapsed_time;    // time consumption
    int first;              // index of the first multiple
    int global_count;       // global prime count
    int low_value;          // local low boundary
    int high_value;         // local high boundary
    register int i;
    int id;                 // process id
    int index;              // index of current prime
    char *marked;           // portion of 2, 3, ..., n
    char *local_prime_list;
    int prime_list_size;
    int n;                  // total problem size (2 - n)
    int p;                  // number of process
    int proc0_size;         // size of subarray on process 0
    int prime;              // current prime
    int size;               // size of 'marked'

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    // check arg
    if (argc != 2) {
        if (!id) printf("Command line: %s <N>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    n = atoi(argv[1]);
    int n_odd = (n - 1) / 2;

    // calculate subarray boundaries
    low_value = (long long)id * n_odd / p + 1;
    high_value = (long long)(id + 1) * n_odd / p;
    size = high_value - low_value + 1;
    // printf("%d: %d, %d, %d\n", id, low_value, high_value, size);

    // make sure all base primes present in process 0
    proc0_size = n_odd / p;
    prime_list_size = ((int)sqrt((double)n) - 1) / 2;

    if ((2 * proc0_size) < (int)sqrt((double)n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    // allocate subarray
    marked = (char*)malloc(size);
    local_prime_list = (char*)malloc(prime_list_size);

    if (marked == NULL || local_prime_list == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (i = 0; i < size; i++) marked[i] = 0;
    for (i = 0; i < prime_list_size; i++) local_prime_list[i] = 0;
    index = 0;
    prime = 3;

    // compute local prime list
    do {
        // low_value for local_prime_list is 1
        // prime * prime > 2 * 1 + 1
        first = (prime * prime - 1) / 2 - 1;
        for (i = first; i < prime_list_size; i += prime) local_prime_list[i] = 1;
        while (local_prime_list[++index]);
        prime = 2 * (index + 1) + 1;
    } while (prime * prime <= sqrt(n));

    index = 0;
    prime = 3;
    // find prime
    do {
        if (prime * prime > (2 * low_value + 1))
            first = (prime * prime - 1) / 2 - low_value;
        else {
            // map low_value to unmarked numbers
            int low_value_prime = 2 * low_value + 1;
            // low_value_prime = k * prime => start from index 0
            if (!(low_value_prime % prime)) first = 0;
            else {
                int k = low_value_prime / prime + 1;
                // k * prime (> low_value_prime) is odd (k is odd), the first number to mark
                if (k & 1)
                    first = (k * prime - 1) / 2 - low_value;
                // k * prime is even, then k * prime + prime is the first number to mark
                else
                    first = ((k + 1) * prime - 1) / 2 - low_value;
            }
        }
        for (i = first; i < size; i += prime) marked[i] = 1;
        while (local_prime_list[++index]);
        prime = 2 * (index + 1) + 1;
    } while (prime * prime <= n);

    // printf("%d: ", id);
    count = 0;
    for (i = 0; i < size; i++){
        // print number and marked flag
        // printf("%d(%d) ", 2 * (low_value + i) + 1, marked[i]);
        if (!marked[i]) count++;
    }
    // printf("\n");

    if (p > 1) MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // for serial baseline (num_proc = 1)
    else global_count = count;

    // stop the timer
    elapsed_time += MPI_Wtime();

    // print result
    if (!id) {
        printf("There are %d primes less than or equal to %d\n", global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    // printf("%d: %10.6f\n", id, elapsed_time);

    MPI_Finalize();
    return 0;
}
