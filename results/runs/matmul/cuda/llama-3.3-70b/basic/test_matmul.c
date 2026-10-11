#include "matmul.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *A, *B, *C;
    A = (double *) malloc(REF_N * REF_N * sizeof(double));
    B = (double *) malloc(REF_N * REF_N * sizeof(double));
    C = (double *) malloc(REF_N * REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            A[i * REF_N + j] = ((i * 3 + j * 7) % 11) + 1;
            B[i * REF_N + j] = ((i * 5 + j * 2) % 13) + 1;
        }
    }
    matmul(A, B, C, REF_N);
    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += C[i * REF_N + j];
        }
    }
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Error: checksum mismatch\n");
        return 1;
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
