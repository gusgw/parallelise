#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "reference.h"

/* Fill functions matching the notebook definitions */
static double A_of(long i, long j)
{
    return (double)(((i * 3 + j * 7) % 11) + 1);
}

static double B_of(long i, long j)
{
    return (double)(((i * 5 + j * 2) % 13) + 1);
}

int main(void)
{
    long n = REF_N;
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double checksum = 0.0;
    double diff;
    int ret = 0;
    long i, j;

    /* Allocate matrices */
    A = (double *)malloc(n * n * sizeof(double));
    B = (double *)malloc(n * n * sizeof(double));
    C = (double *)malloc(n * n * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        return EXIT_FAILURE;
    }

    /* Fill matrices A and B */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Perform matrix multiplication on GPU */
    matmul(A, B, C, n);

    /* Compute checksum */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            checksum += C[i * n + j];
        }
    }

    /* Check against reference */
    diff = fabs(checksum - REF_CHECKSUM);
    if (diff <= REF_TOL) {
        printf("PASS: checksum = %.1f, expected = %.1f, diff = %.6f (within tol %.1f)\n",
               checksum, REF_CHECKSUM, diff, REF_TOL);
        ret = 0;
    } else {
        printf("FAIL: checksum = %.1f, expected = %.1f, diff = %.6f (exceeds tol %.1f)\n",
               checksum, REF_CHECKSUM, diff, REF_TOL);
        ret = 1;
    }

    free(A);
    free(B);
    free(C);

    return ret;
}
