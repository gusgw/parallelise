#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time `iters` Jacobi sweeps on an n x n
 * grid (n = argv[1] default 1024, iters = argv[2] default 120) and print "ranks,time_s"
 * from rank 0. More sweeps => longer, steadier timing (Jacobi is memory-bandwidth-bound,
 * so its speedup is modest and noise-sensitive). bench_sweep.sh runs this at several rank
 * counts and derives the speedup, so a kernel that is not really decomposed shows none. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 120;

    double *u = malloc(sizeof(double) * n * n);
    if (!u) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) u[i * n + j] = u_init(i, j, n);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    jacobi(u, n, iters);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(u);
    MPI_Finalize();
    return 0;
}
