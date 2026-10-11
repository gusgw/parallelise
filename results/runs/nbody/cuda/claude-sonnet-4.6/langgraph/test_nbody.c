/*
 * test_nbody.c  –  Unit test for the GPU nbody_accel kernel.
 *
 * Fills REF_N bodies with the deterministic positions from problem.h,
 * calls nbody_accel (GPU), computes the checksum, and checks it against
 * the notebook oracle (REF_CHECKSUM) within REF_TOL.
 *
 * Build: nvcc -O2 test_nbody.c nbody.o -o test_nbody -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;

    double *px = (double *)malloc(sizeof(double) * n);
    double *py = (double *)malloc(sizeof(double) * n);
    double *ax = (double *)malloc(sizeof(double) * n);
    double *ay = (double *)malloc(sizeof(double) * n);

    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Fill deterministic positions (same formula as the Python notebook). */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Run the GPU kernel. */
    nbody_accel(px, py, ax, ay, n);

    /* Compute the checksum: sum_i ( |ax[i]| + |ay[i]| ). */
    double cs = checksum(ax, ay, n);

    /* Check against the notebook oracle. */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL * REF_CHECKSUM,
                "nbody checksum matches oracle");

    /* Also verify the result is non-trivial (not all zeros). */
    CHECK(cs > 0.0, "checksum is positive");

    free(px); free(py); free(ax); free(ay);
    return TEST_SUMMARY();
}
