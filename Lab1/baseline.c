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
    int i;
    int id;                 // process id
    int index;              // index of current prime
    char *marked;           // portion of 2, 3, ..., n
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

    // calculate subarray boundaries
    low_value = 2 + (long long)id * (n - 1) / p;
    high_value = 1 + (long long)(id + 1) * (n - 1) / p;
    size = high_value - low_value + 1;
    // printf("%d, %d, %d\n", low_value, high_value, size);

    // make sure all base primes present in process 0
    proc0_size = (n - 1) / p;

    if ((2 + proc0_size) < (int)sqrt((double)n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    // allocate subarray
    marked = (char*)malloc(size);

    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (i = 0; i < size; i++) marked[i] = 0;
    if (!id) index = 0;
    prime = 2;

    do {
        if (prime * prime > low_value)
            first = prime * prime - low_value;
        else {
            int r = low_value % prime;
            if (!r) first = 0;
            else first = prime - r;
        }
        for (i = first; i < size; i += prime) marked[i] = 1;
        if (!id) {
            while (marked[++index]);
            prime = index + 2;
        }
        if (p > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= n);

    count = 0;
    for (i = 0; i < size; i++)
        if (!marked[i]) count++;
    if (p > 1) MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // for serial baseline (num_proc = 1)
    else global_count = count;

    // stop the timer
    elapsed_time += MPI_Wtime();

    // print result
    if (!id) {
        printf("There are %d primes less than or equal to %d\n", global_count, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }

    MPI_Finalize();
    return 0;
}
