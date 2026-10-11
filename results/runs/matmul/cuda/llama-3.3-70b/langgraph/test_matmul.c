#include "matmul.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int N = REF_N;
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i * 3 + j * 7) % 11 + 1;
            B[i * N + j] = (i * 5 + j * 2) % 13 + 1;
        }
    }
    matmul(A, B, C, N);
    float checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i * N + j];
        }
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Error: checksum mismatch\n");
        return 1;
    }
    printf("Success: checksum matches\n");
    return 0;
}