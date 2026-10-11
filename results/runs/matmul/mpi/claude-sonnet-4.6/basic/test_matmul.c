/* test_matmul.c -- unit test for the parallel matmul kernel.
 *
 * Every rank calls matmul() the same number of times (required: collective).
 * Only rank 0 checks the result against reference.h and prints the summary.
 * Exit code: 0 = all checks passed, non-zero = at least one failure.
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

    /* Allocate and fill A and B with the deterministic pattern */
    double *A = (double *)malloc(sizeof(double) * n * n);
    double *B = (double *)malloc(sizeof(double) * n * n);
    double *C = (double *)malloc(sizeof(double) * n * n);
    if (!A || !B || !C) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* All ranks call the collective kernel together */
    matmul(A, B, C, n);

    /* Only rank 0 validates and reports */
    if (rank == 0) {
        double cs = checksum(C, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "matmul checksum matches oracle");
        printf("checksum = %.1f  (expected %.1f)\n", cs, REF_CHECKSUM);
        int result = TEST_SUMMARY();
        free(A); free(B); free(C);
        MPI_Finalize();
        return result;
    }

    free(A); free(B); free(C);
    MPI_Finalize();
    return 0;
}
