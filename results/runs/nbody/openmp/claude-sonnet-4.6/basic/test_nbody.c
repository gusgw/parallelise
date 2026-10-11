/* test_nbody.c -- unit test for nbody_accel().
 *
 * Fills REF_N bodies with the deterministic positions from problem.h, runs the
 * kernel, computes the checksum (sum of |ax_i| + |ay_i|), and checks it against
 * the oracle value REF_CHECKSUM within REF_TOL (relative tolerance).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;

    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);

    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "test_nbody: allocation failed\n");
        return 1;
    }

    /* Fill positions with the same deterministic formula as the Python notebook */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Run the kernel under test */
    nbody_accel(px, py, ax, ay, n);

    /* Compute the oracle checksum: sum of |ax_i| + |ay_i| */
    double cs = checksum(ax, ay, n);

    /* Relative error check */
    double rel_err = fabs(cs - REF_CHECKSUM) / fabs(REF_CHECKSUM);

    printf("checksum      = %.10f\n", cs);
    printf("ref checksum  = %.10f\n", REF_CHECKSUM);
    printf("relative error= %.3e  (tol = %.3e)\n", rel_err, REF_TOL);

    if (rel_err > REF_TOL) {
        fprintf(stderr, "FAIL: relative error %.3e exceeds tolerance %.3e\n",
                rel_err, REF_TOL);
        free(px); free(py); free(ax); free(ay);
        return 1;
    }

    printf("PASS\n");
    free(px); free(py); free(ax); free(ay);
    return 0;
}
