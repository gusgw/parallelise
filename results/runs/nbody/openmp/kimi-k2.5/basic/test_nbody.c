#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

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
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

    free(px);
    free(py);
    free(ax);
    free(ay);

    return TEST_SUMMARY();
}
