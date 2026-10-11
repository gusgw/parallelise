#include "matmul.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    double *A = (double *)malloc(REF_N * REF_N * sizeof(double));
    double *B = (double *)malloc(REF_N * REF_N * sizeof(double));
    double *C = (double *)malloc(REF_N * REF_N * sizeof(double));

    // Fill A and B with some values
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            A[i * REF_N + j] = ((i * 3 + j * 7) % 11) + 1;
            B[i * REF_N + j] = ((i * 5 + j * 2) % 13) + 1;
        }
    }

    matmul(A, B, C, REF_N);

    // Calculate checksum
    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += C[i * REF_N + j];
        }
    }

    // Check if checksum is within tolerance
    if (fabs(checksum - REF_CHECKSUM) <= REF_TOL) {
        printf("PASS: checksum = %f\n", checksum);
        return 0;
    } else {
        printf("FAIL: checksum = %f, expected %f\n", checksum, REF_CHECKSUM);
        return 1;
    }
}
