#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nbody.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;
    double *px = malloc(n * sizeof(double));
    double *py = malloc(n * sizeof(double));
    double *ax = malloc(n * sizeof(double));
    double *ay = malloc(n * sizeof(double));

    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    double cs = checksum(ax, ay, n);
    double err = fabs(cs - REF_CHECKSUM);

    printf("checksum = %.16e\n", cs);
    printf("expected = %.16e\n", REF_CHECKSUM);
    printf("error    = %.16e\n", err);
    printf("tol      = %.16e\n", REF_TOL);

    free(px);
    free(py);
    free(ax);
    free(ay);

    if (err <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}
