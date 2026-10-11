#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cg.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time one collective fixed-iteration CG
 * solve (m = argv[1] default 1024, iters = argv[2] default 100) and print a single CSV
 * row "ranks,time_s" from rank 0. The slowest rank's wall time is used. bench_sweep.sh
 * runs this at several rank counts and derives the speedup, so a kernel that is not
 * really decomposed shows no speedup. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long m = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 100;

    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    if (!x || !b) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long t = 0; t < m * m; t++) b[t] = b_of(t);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    cgsolve(x, b, m, iters);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(x); free(b);
    MPI_Finalize();
    return 0;
}
