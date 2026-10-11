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

    for (long i = 0; i < n; i++) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    double res_checksum = checksum(x, m);
    double diff = fabs(res_checksum - REF_CHECKSUM);

    printf("Checksum: %f\n", res_checksum);
    printf("Reference: %f\n", REF_CHECKSUM);
    printf("Diff: %f\n", diff);

    if (diff <= REF_TOL) {
        printf("PASS\n");
        free(x);
        free(b);
        return 0;
    } else {
        printf("FAIL\n");
        free(x);
        free(b);
        return 1;
    }
}
