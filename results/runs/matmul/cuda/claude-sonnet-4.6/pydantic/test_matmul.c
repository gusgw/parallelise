/*
 * test_matmul.c  –  Unit test for the GPU matmul kernel.
 *
 * Fills A and B with the same deterministic integer pattern used in the
 * Python notebook (via problem.h), calls matmul() (GPU), computes the
 * checksum of C, and checks it against the notebook oracle (reference.h)
 * within REF_TOL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    size_t sz = (size_t)n * (size_t)n * sizeof(double);

    double *A = (double *)malloc(sz);
    double *B = (double *)malloc(sz);
    double *C = (double *)malloc(sz);
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Fill inputs with the notebook's deterministic pattern */
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    /* Run the GPU kernel */
    matmul(A, B, C, n);

    /* Compute checksum and compare to oracle */
    double cs = checksum(C, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "matmul checksum");

    free(A);
    free(B);
    free(C);

    return TEST_SUMMARY();
}
