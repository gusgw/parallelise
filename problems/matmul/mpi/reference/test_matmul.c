#include <mpi.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). matmul() is collective, so every rank
 * calls it; only rank 0 checks the checksum against the oracle and reports. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    matmul(A, B, C, n);

    int rc = 0;
    if (rank == 0) {
        double cs = checksum(C, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "matmul checksum matches the oracle");
        rc = TEST_SUMMARY();
    }
    free(A); free(B); free(C);
    MPI_Finalize();
    return rc;
}
