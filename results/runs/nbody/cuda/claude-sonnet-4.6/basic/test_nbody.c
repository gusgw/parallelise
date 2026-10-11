/*
 * test_nbody.c — unit test for nbody_accel (GPU kernel).
 *
 * Fills REF_N bodies with the deterministic positions from problem.h,
 * calls nbody_accel, computes the checksum, and checks it against the
 * notebook oracle (REF_CHECKSUM) within REF_TOL.
 *
 * Plain C, no MPI, no OpenMP.  Build via the project Makefile.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

    /* fill deterministic positions (bit-identical to the Python oracle) */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* run the GPU kernel */
    nbody_accel(px, py, ax, ay, n);

    /* compute checksum: sum_i ( |ax[i]| + |ay[i]| ) */
    double cs = checksum(ax, ay, n);

    /* check against the notebook oracle */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL * REF_CHECKSUM,
                "nbody checksum matches oracle");

    /* also verify the relative error explicitly for a clear diagnostic */
    double rel_err = fabs(cs - REF_CHECKSUM) / REF_CHECKSUM;
    CHECK(rel_err < REF_TOL, "nbody relative error within REF_TOL");

    printf("checksum = %.10f  (oracle %.10f  rel_err %.3e)\n",
           cs, (double)REF_CHECKSUM, rel_err);

    free(px); free(py); free(ax); free(ay);
    return TEST_SUMMARY();
}
