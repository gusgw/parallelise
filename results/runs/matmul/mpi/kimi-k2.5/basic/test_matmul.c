#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matmul.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    /* Allocate matrices */
    A = (double *)malloc(n * n * sizeof(double));
    B = (double *)malloc(n * n * sizeof(double));
    C = (double *)malloc(n * n * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Fill A and B deterministically (all ranks do the same) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Initialize C to zero */
    memset(C, 0, n * n * sizeof(double));

    /* Call the kernel (collective operation) */
    matmul(A, B, C, n);

    /* Compute checksum and verify (only rank 0 checks) */
    if (rank == 0) {
        double cs = checksum(C, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");
        printf("checksum = %.1f (expected %.1f)\n", cs, REF_CHECKSUM);
    }

    free(A);
    free(B);
    free(C);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    MPI_Finalize();
    return result;
}
