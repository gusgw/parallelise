/* test_nbody.c - Unit test for nbody CUDA kernel */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nbody.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;
    double *px = malloc((size_t)n * sizeof(double));
    double *py = malloc((size_t)n * sizeof(double));
    double *ax = malloc((size_t)n * sizeof(double));
    double *ay = malloc((size_t)n * sizeof(double));

    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Fill positions deterministically */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Compute accelerations on GPU */
    nbody_accel(px, py, ax, ay, n);

    /* Compute checksum */
    double cs = checksum(ax, ay, n);

    /* Check against reference */
    double diff = fabs(cs - REF_CHECKSUM);
    printf("Computed checksum: %.16f\n", cs);
    printf("Reference checksum: %.16f\n", REF_CHECKSUM);
    printf("Difference: %.16e\n", diff);
    printf("Tolerance: %.16e\n", REF_TOL);

    free(px);
    free(py);
    free(ax);
    free(ay);

    if (diff <= REF_TOL) {
        printf("PASS: checksum within tolerance\n");
        return 0;
    } else {
        printf("FAIL: checksum difference exceeds tolerance\n");
        return 1;
    }
}
