// test_jacobi.c: unit test for jacobi.cu
#include "jacobi.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>

int main() {
    double u[REF_N * REF_N];
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = (i == 0) ? 1.0 : 0.0;
        }
    }
    jacobi(u, REF_N, REF_ITERS);
    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += u[i * REF_N + j];
        }
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}