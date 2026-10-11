#include <mpi.h>
#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). jacobi() is collective, so every rank
 * calls it; only rank 0 checks the final grid's checksum against the oracle. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *u = malloc(sizeof(double) * n * n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) u[i * n + j] = u_init(i, j, n);

    jacobi(u, n, REF_ITERS);

    int rc = 0;
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "jacobi checksum matches the oracle");
        rc = TEST_SUMMARY();
    }
    free(u);
    MPI_Finalize();
    return rc;
}
