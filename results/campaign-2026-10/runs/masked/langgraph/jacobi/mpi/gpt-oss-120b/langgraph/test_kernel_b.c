#include <mpi.h>
#include <stdlib.h>
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long N = REF_N;
    int ITERS = (int)REF_ITERS;
    double *u = (double *)malloc(N * N * sizeof(double));
    if (!u) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long i = 0; i < N * N; ++i) u[i] = 0.0;

    kernel_b(u, N, ITERS);
    MPI_Barrier(MPI_COMM_WORLD);
    double chk = checksum(u, N);
    if (rank == 0) {
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    }
    free(u);
    MPI_Finalize();
    return TEST_SUMMARY();
}
