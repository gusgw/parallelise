#include "jacobi.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *u = (double *) malloc(REF_N * REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = (i + j) * 0.1;
        }
    }
    jacobi(u, REF_N, REF_ITERS);
    double checksum = 0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += u[i * REF_N + j];
        }
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Checksum mismatch: expected %f, got %f\n", REF_CHECKSUM, checksum);
        return 1;
    }
    printf("Checksum match: %f\n", checksum);
    free(u);
    return 0;
}
