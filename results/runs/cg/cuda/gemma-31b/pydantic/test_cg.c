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
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    double res_checksum = checksum(x, m);
    printf("Result checksum: %f\n", res_checksum);
    printf("Reference checksum: %f\n", REF_CHECKSUM);

    if (fabs(res_checksum - REF_CHECKSUM) < REF_TOL) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        free(b);
        free(x);
        return 1;
    }

    free(b);
    free(x);
    return 0;
}
