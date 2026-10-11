#include "kernel_d.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *x = (double *)malloc(REF_M * sizeof(double));
    double *b = (double *)malloc(REF_M * sizeof(double));

    for (long i = 0; i < REF_M; i++) {
        b[i] = (i + 1) * (i + 2);
    }

    cgsolve(x, b, REF_M, REF_ITERS);

    double checksum = 0.0;
    for (long i = 0; i < REF_M; i++) {
        checksum += x[i];
    }

    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed: checksum = %f, expected %f\n", checksum, REF_CHECKSUM);
        return 1;
    }

    printf("Test passed\n");
    return 0;
}
