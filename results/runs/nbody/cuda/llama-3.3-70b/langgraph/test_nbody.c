#include "nbody.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int n = REF_N;
    double *px = (double *)malloc(n * sizeof(double));
    double *py = (double *)malloc(n * sizeof(double));
    double *ax = (double *)malloc(n * sizeof(double));
    double *ay = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }
    nbody_accel(px, py, ax, ay, n);
    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += fabs(ax[i]) + fabs(ay[i]);
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Checksum mismatch: %f != %f\n", checksum, REF_CHECKSUM);
        return 1;
    }
    printf("Checksum match: %f\n", checksum);
    free(px);
    free(py);
    free(ax);
    free(ay);
    return 0;
}