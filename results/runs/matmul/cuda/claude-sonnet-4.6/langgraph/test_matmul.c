/*
 * test_matmul.c  –  Unit test for the GPU matmul kernel.
 *
 * Fills A and B with the same deterministic integer pattern used by the Python
 * notebook, calls matmul() (which runs on the GPU), computes the checksum of C,
 * and checks it against the notebook oracle stored in reference.h.
 *
 * Plain C (no MPI, no OpenMP).  Build via the project Makefile:
 *     make test_matmul
 *     ./test_matmul
 */

#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    /* Fill A and B with the deterministic pattern from problem.h / the notebook. */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Run the GPU kernel (copies to/from device internally). */
    matmul(A, B, C, n);

    /* Compute the checksum of C and compare to the notebook oracle. */
    double cs = checksum(C, n);

    printf("checksum = %.12g  (expected %.12g)\n", cs, REF_CHECKSUM);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "matmul checksum matches notebook oracle");

    free(A);
    free(B);
    free(C);

    return TEST_SUMMARY();
}
