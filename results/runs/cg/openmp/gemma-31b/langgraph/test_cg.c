#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"

int main() {
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));

    if (!x || !b) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (long i = 0; i < n; i++) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    double res = checksum(x, m);
    printf("Checksum: %f\n", res);
    printf("Reference: %f\n", REF_CHECKSUM);

    if (fabs(res - REF_CHECKSUM) > REF_TOL) {
        printf("FAILED: Difference %f exceeds tolerance %f\n", fabs(res - REF_CHECKSUM), REF_TOL);
        free(x);
        free(b);
        return 1;
    }

    printf("PASSED\n");
    free(x);
    free(b);
    return 0;
}
