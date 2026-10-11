#include <mpi.h>
#include <stdlib.h>
#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *u = (double *)malloc(n * n * sizeof(double));
    if (!u) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Initialise the grid using the problem helper */
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, (int)REF_ITERS);

    double cs = checksum(u, n);
    if (rank == 0) {
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(u);
    MPI_Finalize();
    return TEST_SUMMARY();
}
