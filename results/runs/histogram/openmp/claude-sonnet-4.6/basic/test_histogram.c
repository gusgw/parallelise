/* test_histogram.c — unit test for the histogram kernel.
 *
 * Checks:
 *   1. The per-bin counts sum to exactly REF_N (no values lost or double-counted).
 *   2. The weighted checksum matches REF_CHECKSUM within REF_TOL.
 *
 * The input is generated with x_of() from problem.h, which is bit-identical to
 * the Python oracle in histogram.ipynb.
 */

#include <stdlib.h>
#include <stdio.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n     = REF_N;
    int  nbins = (int)REF_NBINS;

    /* Allocate and fill the input array with the deterministic pseudo-random
     * values defined in problem.h (identical to the Python oracle). */
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    if (!x) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    /* Allocate the output counts array. */
    long *counts = (long *)malloc(sizeof(long) * (size_t)nbins);
    if (!counts) { fprintf(stderr, "malloc failed\n"); free(x); return 1; }

    /* Run the kernel under test. */
    histogram(x, n, counts, nbins);

    /* --- Check 1: total count equals n ------------------------------------ */
    long total = 0;
    for (int b = 0; b < nbins; b++) total += counts[b];
    CHECK(total == n, "total count == REF_N");

    /* --- Check 2: weighted checksum matches the oracle -------------------- */
    double cs = checksum(counts, nbins);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches REF_CHECKSUM");

    free(x);
    free(counts);
    return TEST_SUMMARY();
}
