#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Allocate and initialize grid */
    long n = REF_N;
    double *u = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run the kernel */
    jacobi(u, n, (int)REF_ITERS);

    /* Only rank 0 checks and reports */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "jacobi checksum");
    }

    free(u);

    MPI_Finalize();

    /* Only rank 0 reports test summary */
    if (rank == 0) {
        return TEST_SUMMARY();
    }
    return 0;
}
