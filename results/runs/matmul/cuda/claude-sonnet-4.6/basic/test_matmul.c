/*
 * test_matmul.c  --  Unit test for the GPU matmul kernel.
 *
 * Fills A and B with the same deterministic integer pattern used by the Python
 * notebook, calls matmul() (GPU), computes the checksum of C, and checks it
 * against the oracle value from reference.h within REF_TOL.
 *
 * Plain C (no MPI, no OpenMP).  Build via the project Makefile:
 *     make test_matmul
 * Run:
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
        fprintf(stderr, "test_matmul: malloc failed\n");
        return 1;
    }

    /* Fill inputs with the deterministic pattern from problem.h / the notebook. */
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    /* Run the GPU kernel. */
    matmul(A, B, C, n);

    /* Compute checksum and compare to the oracle. */
    double cs = checksum(C, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "matmul checksum matches notebook oracle");

    free(A);
    free(B);
    free(C);
    return TEST_SUMMARY();
}
