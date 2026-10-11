/* test_histogram.c — unit tests for the histogram kernel.
 *
 * Checks:
 *   1. All counts are non-negative.
 *   2. The counts sum exactly to n (no element lost or double-counted).
 *   3. The weighted checksum matches the Python oracle within REF_TOL.
 *   4. A small, hand-verifiable case (n=4, nbins=4) with known bin assignments.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* Test 1 – small, hand-verifiable case                                */
    /* ------------------------------------------------------------------ */
    {
        /* Four values spread evenly across [0,1): 0.125, 0.375, 0.625, 0.875
         * With nbins=4 each falls in its own bin (bins 0,1,2,3). */
        const double xsmall[] = {0.125, 0.375, 0.625, 0.875};
        long csmall[4];
        histogram(xsmall, 4, csmall, 4);

        long total_small = 0;
        for (int b = 0; b < 4; b++) total_small += csmall[b];
        CHECK(total_small == 4, "small: total count == n");
        CHECK(csmall[0] == 1, "small: bin 0 has 1 element");
        CHECK(csmall[1] == 1, "small: bin 1 has 1 element");
        CHECK(csmall[2] == 1, "small: bin 2 has 1 element");
        CHECK(csmall[3] == 1, "small: bin 3 has 1 element");
    }

    /* ------------------------------------------------------------------ */
    /* Test 2 – clamping: a value of exactly 1.0 must land in the top bin  */
    /* ------------------------------------------------------------------ */
    {
        /* x_of(i) is always < 1.0 by construction, so we inject 1.0 manually. */
        const double xclamp[] = {1.0, 0.0};
        long cclamp[4];
        histogram(xclamp, 2, cclamp, 4);
        CHECK(cclamp[3] == 1, "clamp: value 1.0 lands in top bin");
        CHECK(cclamp[0] == 1, "clamp: value 0.0 lands in bin 0");
    }

    /* ------------------------------------------------------------------ */
    /* Test 3 – full oracle run (REF_N elements, REF_NBINS bins)           */
    /* ------------------------------------------------------------------ */
    {
        long n     = REF_N;
        int  nbins = (int)REF_NBINS;

        double *x = (double *)malloc(sizeof(double) * (size_t)n);
        long   *c = (long   *)malloc(sizeof(long)   * (size_t)nbins);
        if (!x || !c) {
            fprintf(stderr, "allocation failed\n");
            free(x); free(c);
            return 1;
        }

        /* Fill with the same deterministic pseudo-random sequence as the notebook. */
        for (long i = 0; i < n; i++) x[i] = x_of(i);

        histogram(x, n, c, nbins);

        /* 3a – all counts non-negative */
        int any_negative = 0;
        for (int b = 0; b < nbins; b++) if (c[b] < 0) any_negative = 1;
        CHECK(!any_negative, "oracle: all bin counts >= 0");

        /* 3b – counts sum to n */
        long total = 0;
        for (int b = 0; b < nbins; b++) total += c[b];
        CHECK(total == n, "oracle: sum of counts == n");

        /* 3c – weighted checksum matches the Python oracle */
        double cs = checksum(c, nbins);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle: checksum matches reference");

        free(x);
        free(c);
    }

    return TEST_SUMMARY();
}
