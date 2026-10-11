#include "jacobi.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *u = malloc(REF_N * REF_N * sizeof(double));
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
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    printf("PASSED\n");
    free(u);
    return 0;
}
