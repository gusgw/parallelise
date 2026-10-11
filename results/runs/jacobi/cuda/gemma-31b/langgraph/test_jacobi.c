#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    int iters = (int)REF_ITERS;
    double *u = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, iters);

    double result = checksum(u, n);
    printf("Checksum: %f\n", result);
    printf("Reference: %f\n", REF_CHECKSUM);

    if (fabs(result - REF_CHECKSUM) > REF_TOL) {
        printf("FAILED: Difference %f exceeds tolerance %f\n", fabs(result - REF_CHECKSUM), REF_TOL);
        free(u);
        return 1;
    }

    printf("PASSED\n");
    free(u);
    return 0;
}
