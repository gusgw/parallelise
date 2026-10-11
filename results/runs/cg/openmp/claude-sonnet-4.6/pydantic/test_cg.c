/* test_cg.c — unit test for cgsolve against the Python-notebook oracle.
 *
 * Builds the same deterministic RHS used by the notebook (via b_of from
 * problem.h), runs cgsolve, and checks the checksum against REF_CHECKSUM
 * within REF_TOL (both from reference.h).
 */

#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m     = REF_M;
    int  iters = (int)REF_ITERS;
    long n     = m * m;

    double *b = (double *)malloc(sizeof(double) * (size_t)n);
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    if (!b || !x) {
        printf("FAIL: allocation failed\n");
        return 1;
    }

    /* Fill the deterministic RHS exactly as the notebook does. */
    for (long idx = 0; idx < n; idx++) b[idx] = b_of(idx);

    /* Run the solver. */
    cgsolve(x, b, m, iters);

    /* Compute the checksum and compare to the oracle. */
    double cs = checksum(x, m);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "cgsolve checksum matches oracle");

    free(b);
    free(x);
    return TEST_SUMMARY();
}
