#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *x = (double *)malloc(REF_M * REF_M * sizeof(double));
    double *b = (double *)malloc(REF_M * REF_M * sizeof(double));
    for (long idx = 0; idx < REF_M * REF_M; idx++) {
        b[idx] = ((idx * 97) % 211) / 211.0;
    }
    cgsolve(x, b, REF_M, REF_ITERS);
    double checksum = 0.0;
    for (long idx = 0; idx < REF_M * REF_M; idx++) {
        checksum += x[idx];
    }
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    printf("OK\n");
    free(x);
    free(b);
    return 0;
}