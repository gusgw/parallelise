#include "jacobi.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *u = malloc(REF_N * REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = i + j;
        }
    }
    jacobi(u, REF_N, REF_ITERS);
    double checksum = 0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += u[i * REF_N + j];
        }
    }
    printf("Checksum: %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    } else {
        printf("Test passed!\n");
        return 0;
    }
}
