#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "integrate.h"

/* Time one distributed run and print a single CSV row "ranks,time_s" (rank 0).
 * The rank sweep is done by bench_sweep.sh, which runs this binary at several rank
 * counts and assembles the scaling table. n is argv[1] (default 2e8). Time is the
 * slowest rank's wall time. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 200000000L;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    volatile double r = integrate(0.0, 1.0, n);
    double elapsed = MPI_Wtime() - start;
    (void)r;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%d,%.6f\n", size, slowest);
    }
    MPI_Finalize();
    return 0;
}
