#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this): time one collective matmul at size n
 * (argv[1], default 1024) and print a single CSV row "ranks,time_s" from rank 0. The
 * slowest rank's wall time is used. bench_sweep.sh runs this at several rank counts and
 * derives the speedup, so a routine that is not really decomposed shows no speedup. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    /* One untimed warm-up run, so that first-write page faults and other one-time
     * costs are paid before the timed run begins. Otherwise the one-process
     * baseline is charged for them and every speed-up measured against it is
     * inflated. */
    matmul(A, B, C, n);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    matmul(A, B, C, n);
    double elapsed = MPI_Wtime() - t0;

    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d,%.6f\n", size, slowest);

    free(A); free(B); free(C);
    MPI_Finalize();
    return 0;
}
