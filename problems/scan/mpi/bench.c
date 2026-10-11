#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "scan.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time one collective scan at length n
 * (argv[1], default 134217728) and print a single CSV row "ranks,time_s" from rank 0.
 * The slowest rank's wall time is used. bench_sweep.sh runs this at several rank counts
 * and derives the speedup, so a kernel that is not really decomposed shows no speedup. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 134217728L;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!in || !out) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long i = 0; i < n; i++) in[i] = in_of(i);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    scan(in, out, n);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(in); free(out);
    MPI_Finalize();
    return 0;
}
