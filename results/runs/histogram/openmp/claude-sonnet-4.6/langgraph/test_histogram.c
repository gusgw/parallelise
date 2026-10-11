/* test_histogram.c — unit tests for the histogram kernel.
 *
 * Tests performed:
 *   1. Tiny hand-checkable case (n=4, nbins=4).
 *   2. Full oracle case: REF_N values, REF_NBINS bins.
 *      a. Total count == REF_N.
 *      b. No bin is negative.
 *      c. Weighted checksum matches REF_CHECKSUM within REF_TOL.
 *   3. Single-element edge case.
 *   4. All values map to the same bin (stress the atomic merge).
 */

#include <stdlib.h>
#include <string.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

static double checksum_d(const long *counts, int nbins)
{
    return checksum(counts, nbins);
}

/* ------------------------------------------------------------------ tests   */

static void test_tiny(void)
{
    /* 4 values spread evenly across [0,1): each should land in its own bin. */
    double x[4] = {0.1, 0.35, 0.6, 0.85};
    long counts[4];
    histogram(x, 4, counts, 4);

    /* bin 0: [0.00, 0.25) -> 0.1          -> counts[0] == 1 */
    /* bin 1: [0.25, 0.50) -> 0.35         -> counts[1] == 1 */
    /* bin 2: [0.50, 0.75) -> 0.6          -> counts[2] == 1 */
    /* bin 3: [0.75, 1.00) -> 0.85         -> counts[3] == 1 */
    CHECK(counts[0] == 1, "tiny: bin 0 count");
    CHECK(counts[1] == 1, "tiny: bin 1 count");
    CHECK(counts[2] == 1, "tiny: bin 2 count");
    CHECK(counts[3] == 1, "tiny: bin 3 count");

    long total = counts[0] + counts[1] + counts[2] + counts[3];
    CHECK(total == 4, "tiny: total count");
}

static void test_zeros_output(void)
{
    /* histogram() must zero counts before filling, even if they were dirty. */
    long counts[4] = {99, 99, 99, 99};
    double x[2] = {0.1, 0.9};
    histogram(x, 2, counts, 4);
    long total = counts[0] + counts[1] + counts[2] + counts[3];
    CHECK(total == 2, "zero-init: total count");
}

static void test_clamp(void)
{
    /* A value of exactly 1.0 would give bin == nbins; must be clamped. */
    /* We can't easily produce 1.0 from x_of, so construct it directly.  */
    double x[1] = {1.0};
    long counts[4] = {0, 0, 0, 0};
    histogram(x, 1, counts, 4);
    long total = counts[0] + counts[1] + counts[2] + counts[3];
    CHECK(total == 1, "clamp: total count == 1");
    CHECK(counts[3] == 1, "clamp: value 1.0 lands in last bin");
}

static void test_single_bin(void)
{
    /* All values in [0, 0.25) -> bin 0 only. */
    double x[5] = {0.0, 0.1, 0.2, 0.24, 0.01};
    long counts[4];
    histogram(x, 5, counts, 4);
    CHECK(counts[0] == 5, "single-bin: all in bin 0");
    CHECK(counts[1] == 0, "single-bin: bin 1 empty");
    CHECK(counts[2] == 0, "single-bin: bin 2 empty");
    CHECK(counts[3] == 0, "single-bin: bin 3 empty");
}

static void test_oracle(void)
{
    long n     = REF_N;
    int  nbins = (int)REF_NBINS;

    double *x      = (double *)malloc(sizeof(double) * (size_t)n);
    long   *counts = (long   *)malloc(sizeof(long)   * (size_t)nbins);
    if (!x || !counts) {
        printf("SKIP: oracle test — allocation failed\n");
        free(x); free(counts);
        return;
    }

    /* Fill with the deterministic pseudo-random sequence from problem.h. */
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    histogram(x, n, counts, nbins);

    /* (a) Total count must equal n. */
    long total = 0;
    for (int b = 0; b < nbins; b++) total += counts[b];
    CHECK(total == n, "oracle: total count == REF_N");

    /* (b) No bin may be negative. */
    int any_negative = 0;
    for (int b = 0; b < nbins; b++) if (counts[b] < 0) any_negative = 1;
    CHECK(!any_negative, "oracle: no negative bin count");

    /* (c) Weighted checksum must match the Python oracle within REF_TOL. */
    double cs = checksum_d(counts, nbins);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle: checksum matches REF_CHECKSUM");

    free(x);
    free(counts);
}

/* ------------------------------------------------------------------ main    */

int main(void)
{
    test_tiny();
    test_zeros_output();
    test_clamp();
    test_single_bin();
    test_oracle();

    return TEST_SUMMARY();
}
