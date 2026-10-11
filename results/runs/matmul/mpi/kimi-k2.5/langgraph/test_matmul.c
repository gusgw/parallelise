#include <mpi.h>
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

    /* Allocate matrices */
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    if (!A || !B || !C) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Fill A and B deterministically (same on all ranks) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Call the collective kernel (all ranks must call) */
    matmul(A, B, C, n);

    /* Compute checksum and verify (only rank 0 asserts) */
    if (rank == 0) {
        double cs = checksum(C, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");
        printf("checksum = %.1f (expected %.1f)\n", cs, REF_CHECKSUM);
        int result = TEST_SUMMARY();
        free(A);
        free(B);
        free(C);
        MPI_Finalize();
        return result;
    }

    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}
