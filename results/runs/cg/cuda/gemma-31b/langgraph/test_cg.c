#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"

int main() {
    long m = REF_M;
    int iters = REF_ITERS;
    long n = m * m;
    double *b = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    double res = checksum(x, m);
    printf("Checksum: %f\n", res);
    printf("Reference: %f\n", REF_CHECKSUM);
    printf("Diff: %f\n", fabs(res - REF_CHECKSUM));

    if (fabs(res - REF_CHECKSUM) <= REF_TOL) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        return 1;
    }

    free(b);
    free(x);
    return 0;
}
