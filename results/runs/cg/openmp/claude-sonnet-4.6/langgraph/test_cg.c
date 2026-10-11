/* test_cg.c — unit tests for cgsolve() against the Python-notebook oracle.
 *
 * Checks:
 *   1. The checksum of the 64×64 solution after 100 iterations matches the
 *      reference value within REF_TOL (the primary correctness gate).
 *   2. The solver is deterministic: two back-to-back calls on the same input
 *      produce bit-identical results.
 *   3. A trivial 1×1 system (A = [[4]], b = [1]) converges in one iteration
 *      to x = 0.25.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* Test 1: checksum matches the notebook oracle                        */
    /* ------------------------------------------------------------------ */
    {
        long m     = REF_M;
        int  iters = (int)REF_ITERS;
        long n     = m * m;

        double *b = (double *)malloc(sizeof(double) * (size_t)n);
        double *x = (double *)malloc(sizeof(double) * (size_t)n);

        for (long idx = 0; idx < n; idx++) b[idx] = b_of(idx);

        cgsolve(x, b, m, iters);

        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "checksum matches notebook oracle");

        free(b);
        free(x);
    }

    /* ------------------------------------------------------------------ */
    /* Test 2: determinism — two calls give identical results              */
    /* ------------------------------------------------------------------ */
    {
        long m     = REF_M;
        int  iters = (int)REF_ITERS;
        long n     = m * m;

        double *b  = (double *)malloc(sizeof(double) * (size_t)n);
        double *x1 = (double *)malloc(sizeof(double) * (size_t)n);
        double *x2 = (double *)malloc(sizeof(double) * (size_t)n);

        for (long idx = 0; idx < n; idx++) b[idx] = b_of(idx);

        cgsolve(x1, b, m, iters);
        cgsolve(x2, b, m, iters);

        int identical = 1;
        for (long k = 0; k < n; k++) {
            if (x1[k] != x2[k]) { identical = 0; break; }
        }
        CHECK(identical, "two calls with same input give identical results");

        free(b); free(x1); free(x2);
    }

    /* ------------------------------------------------------------------ */
    /* Test 3: 1×1 system — A = [[4]], b = [1], solution x = [0.25]       */
    /* ------------------------------------------------------------------ */
    {
        double b1 = 1.0;
        double x1 = 0.0;
        cgsolve(&x1, &b1, 1L, 1);
        CHECK_CLOSE(x1, 0.25, 1e-12, "1x1 system: x = b/4 = 0.25");
    }

    /* ------------------------------------------------------------------ */
    /* Test 4: zero right-hand side — solution must stay zero              */
    /* ------------------------------------------------------------------ */
    {
        long m = 8L;
        long n = m * m;
        double *b = (double *)calloc((size_t)n, sizeof(double));
        double *x = (double *)malloc(sizeof(double) * (size_t)n);

        cgsolve(x, b, m, 10);

        double cs = checksum(x, m);
        CHECK_CLOSE(cs, 0.0, 1e-12, "zero rhs gives zero solution");

        free(b); free(x);
    }

    return TEST_SUMMARY();
}
