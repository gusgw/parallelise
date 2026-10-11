#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time one collective acceleration step
 * for n bodies (argv[1], default 32768) and print a single CSV row "ranks,time_s" from
 * rank 0. The slowest rank's wall time is used. bench_sweep.sh runs this at several
 * rank counts and derives the speedup, so a kernel that is not really decomposed shows
 * no speedup. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 32768L;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    if (!px || !py || !ax || !ay) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    nbody_accel(px, py, ax, ay, n);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(px); free(py); free(ax); free(ay);
    MPI_Finalize();
    return 0;
}
