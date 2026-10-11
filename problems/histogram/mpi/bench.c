#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "histogram.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time one collective histogram of n
 * values (argv[1], default 200000000 -- a 1.6 GB input array per rank) into nbins bins
 * (argv[2], default 256) and print a single CSV row "ranks,time_s" from rank 0. The
 * slowest rank's wall time is used. bench_sweep.sh runs this at several rank counts and
 * derives the speedup, so a kernel that is not really decomposed shows no speedup. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 200000000L;
    int nbins = (argc > 2) ? atoi(argv[2]) : 256;
    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);
    if (!x || !counts) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    histogram(x, n, counts, nbins);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(x); free(counts);
    MPI_Finalize();
    return 0;
}
