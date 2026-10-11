#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nbody.h"
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    double *px = malloc(n * sizeof(double));
    double *py = malloc(n * sizeof(double));
    double *ax = malloc(n * sizeof(double));
    double *ay = malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    double res = checksum(ax, ay, n);
    printf("Checksum: %.10f\n", res);
    printf("Reference: %.10f\n", REF_CHECKSUM);
    printf("Diff: %.10e\n", fabs(res - REF_CHECKSUM));

    if (fabs(res - REF_CHECKSUM) <= REF_TOL) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        free(px); free(py); free(ax); free(ay);
        return 1;
    }

    free(px); free(py); free(ax); free(ay);
    return 0;
}
