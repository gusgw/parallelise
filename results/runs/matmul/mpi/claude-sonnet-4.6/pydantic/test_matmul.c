/* test_matmul.c — unit test for the parallel matmul kernel.
 *
 * Every MPI rank calls matmul() the same number of times (one call) so that
 * the collective inside the kernel does not deadlock.  Only rank 0 checks the
 * result against the reference oracle and prints the test summary.
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    double *A = (double *)malloc(sizeof(double) * (size_t)(n * n));
    double *B = (double *)malloc(sizeof(double) * (size_t)(n * n));
    double *C = (double *)malloc(sizeof(double) * (size_t)(n * n));
    if (!A || !B || !C) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill A and B with the deterministic values from problem.h */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Collective call — every rank participates */
    matmul(A, B, C, n);

    /* Only rank 0 validates and reports */
    int result = 0;
    if (rank == 0) {
        double cs = checksum(C, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "matmul checksum matches oracle");
        result = TEST_SUMMARY();
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return result;
}
