#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    matmul(A, B, C, n);

    double result_checksum = checksum(C, n);
    double diff = fabs(result_checksum - REF_CHECKSUM);

    printf("Expected checksum: %f\n", REF_CHECKSUM);
    printf("Actual checksum:   %f\n", result_checksum);
    printf("Difference:        %f\n", diff);

    if (diff <= REF_TOL) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        free(A); free(B); free(C);
        return 1;
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
