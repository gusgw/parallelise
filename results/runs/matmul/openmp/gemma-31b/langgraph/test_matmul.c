#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    matmul(A, B, C, n);

    double actual_checksum = checksum(C, n);
    printf("Actual checksum: %f\n", actual_checksum);
    printf("Expected checksum: %f\n", REF_CHECKSUM);

    if (fabs(actual_checksum - REF_CHECKSUM) <= REF_TOL) {
        printf("PASS\n");
        free(A);
        free(B);
        free(C);
        return 0;
    } else {
        printf("FAIL\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }
}
