#include "jacobi.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    double *u = (double *)malloc(REF_N * REF_N * sizeof(double));
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
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Error: checksum differs from reference by more than REF_TOL\n");
        return 1;
    }
    free(u);
    return 0;
}