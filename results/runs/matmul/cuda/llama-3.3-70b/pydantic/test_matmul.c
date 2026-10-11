// test_matmul.c: unit test for matmul.cu
#include "matmul.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h> // include stdlib.h for malloc
#include <math.h>

int main() {
    // allocate host buffers
    double *A, *B, *C;
    A = (double *)malloc(REF_N * REF_N * sizeof(double));
    B = (double *)malloc(REF_N * REF_N * sizeof(double));
    C = (double *)malloc(REF_N * REF_N * sizeof(double));

    // fill A and B with deterministic values
    for (int i = 0; i < REF_N; i++) {
        for (int j = 0; j < REF_N; j++) {
            A[i * REF_N + j] = ((i * 3 + j * 7) % 11) + 1;
            B[i * REF_N + j] = ((i * 5 + j * 2) % 13) + 1;
        }
    }

    // call matmul
    matmul(A, B, C, REF_N);

    // compute checksum
    double checksum = 0.0;
    for (int i = 0; i < REF_N; i++) {
        for (int j = 0; j < REF_N; j++) {
            checksum += C[i * REF_N + j];
        }
    }

    // check against reference
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }

    printf("OK\n");
    return 0;
}