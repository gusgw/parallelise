#include <stdlib.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the check
 * checks it at 1 thread and at many -- the checksum must be the same. */
int main(void)
{
    long n = REF_N;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);
    CHECK_CLOSE(checksum(ax, ay, n), REF_CHECKSUM, REF_TOL,
                "nbody acceleration checksum matches the oracle");

    free(px); free(py); free(ax); free(ay);
    return TEST_SUMMARY();
}
