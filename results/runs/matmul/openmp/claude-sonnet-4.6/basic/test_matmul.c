/* test_matmul.c -- Unit test for matmul().
 * Fills A and B with the same deterministic integer values used by the Python
 * notebook, runs matmul(), computes the checksum of C, and checks it against
 * the oracle value from reference.h within REF_TOL. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;

    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    if (!A || !B || !C) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Fill A and B with the same deterministic values as the notebook. */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Run the kernel under test. */
    matmul(A, B, C, n);

    /* Compute the checksum (sum of all entries of C). */
    double cs = checksum(C, n);

    /* Validate against the oracle. */
    double err = fabs(cs - REF_CHECKSUM);
    printf("checksum = %.1f  expected = %.1f  err = %.6g  tol = %.1f\n",
           cs, REF_CHECKSUM, err, REF_TOL);

    if (err > REF_TOL) {
        printf("FAIL: checksum error %.6g exceeds tolerance %.1f\n", err, REF_TOL);
        free(A); free(B); free(C);
        return 1;
    }

    printf("PASS\n");
    free(A); free(B); free(C);
    return 0;
}
