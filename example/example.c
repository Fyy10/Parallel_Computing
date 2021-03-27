#include "mpi.h"
#include "stdio.h"
#include "math.h"

int main(int argc, char *argv[]) {
    int done = 0, n, myid, numprocs, i;
    double mypi, pi, sum;
    double startwtime, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    // printf("%d\n", MPI_MAX_PROCESSOR_NAME);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stderr, "Process %d on %s\n", myid, processor_name);
    fflush(stderr);

    n = 0;
    while (!done) {
        if (myid == 0) {
            printf("Input a number less than 100000000 (input 0 to exit): ");
            fflush(stdout);
            scanf("%d", &n);
            startwtime = MPI_Wtime();
        }
        // broadcast n
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0)
            done = 1;
        else {
            sum = 0.0;
            for (i = myid + 1; i <= n; i+= numprocs) {
                sum += i;
            }
            // partial sum of each process
            mypi = sum;
            MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                // process 0 print the approximate value
                printf("Result: %.16f\n", pi);
                endwtime = MPI_Wtime();
                printf("Time use: %f\n", endwtime - startwtime);
            }
        }
    }
    MPI_Finalize();
    return 0;
}
