#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    int iters = REF_ITERS;
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

    if (fabs(result - REF_CHECKSUM) < REF_TOL) {
        printf("PASS\n");
        free(u);
        return 0;
    } else {
        printf("FAIL\n");
        free(u);
        return 1;
    }
}
