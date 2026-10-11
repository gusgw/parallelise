#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nbody.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    double *px = (double *)malloc(n * sizeof(double));
    double *py = (double *)malloc(n * sizeof(double));
    double *ax = (double *)malloc(n * sizeof(double));
    double *ay = (double *)malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    double result = checksum(ax, ay, n);
    printf("Checksum: %f\n", result);
    printf("Reference: %f\n", REF_CHECKSUM);

    if (fabs(result - REF_CHECKSUM) < REF_TOL) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        free(px);
        free(py);
        free(ax);
        free(ay);
        return 1;
    }

    free(px);
    free(py);
    free(ax);
    free(ay);
    return 0;
}
