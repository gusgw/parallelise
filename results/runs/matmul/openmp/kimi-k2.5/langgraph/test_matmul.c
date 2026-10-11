#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Fill A and B using the deterministic fill functions */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Compute C = A * B */
    matmul(A, B, C, n);

    /* Compute checksum and compare with reference */
    double cs = checksum(C, n);
    double diff = fabs(cs - REF_CHECKSUM);

    int pass = (diff <= REF_TOL);

    printf("checksum = %.1f (expected %.1f, diff = %g, tol = %g)\n",
           cs, REF_CHECKSUM, diff, REF_TOL);
    printf("TEST %s\n", pass ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);

    return pass ? 0 : 1;
}
