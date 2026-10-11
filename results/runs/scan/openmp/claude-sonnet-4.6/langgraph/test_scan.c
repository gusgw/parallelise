/* test_scan.c — unit tests for the parallel inclusive prefix-sum kernel.
 *
 * Tests performed:
 *   1. Tiny known-answer test (n=5, hand-computed values).
 *   2. Single-element edge case (n=1).
 *   3. Two-element edge case (n=2).
 *   4. Full oracle check: fill REF_N elements with in_of(), run scan(),
 *      compute checksum(), compare to REF_CHECKSUM within REF_TOL.
 *      Executed at whatever OMP_NUM_THREADS is set to, so the Makefile
 *      'check' target exercises both 1-thread and nproc-thread runs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ */
/* Helper: allocate, fill with in_of(), run scan, return checksum.    */
/* ------------------------------------------------------------------ */
static double run_checksum(long n)
{
    double *in  = (double *)malloc(sizeof(double) * (size_t)n);
    double *out = (double *)malloc(sizeof(double) * (size_t)n);
    if (!in || !out) {
        fprintf(stderr, "allocation failed for n=%ld\n", n);
        free(in); free(out);
        return 0.0;
    }
    for (long i = 0; i < n; i++) in[i] = in_of(i);
    scan(in, out, n);
    double cs = checksum(out, n);
    free(in);
    free(out);
    return cs;
}

/* ------------------------------------------------------------------ */
int main(void)
{
    /* ---- Test 1: tiny hand-computed inclusive scan ------------------ */
    {
        double in[5]       = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        double out[5]      = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        double expected[5] = { 1.0, 3.0, 6.0, 10.0, 15.0 };

        scan(in, out, 5);

        for (int i = 0; i < 5; i++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "tiny scan out[%d]", i);
            CHECK_CLOSE(out[i], expected[i], 1e-12, msg);
        }
    }

    /* ---- Test 2: single-element edge case -------------------------- */
    {
        double in[1]  = { 3.14 };
        double out[1] = { 0.0  };
        scan(in, out, 1);
        CHECK_CLOSE(out[0], 3.14, 1e-12, "single-element scan");
    }

    /* ---- Test 3: two-element edge case ----------------------------- */
    {
        double in[2]  = { 1.5, 2.5 };
        double out[2] = { 0.0, 0.0 };
        scan(in, out, 2);
        CHECK_CLOSE(out[0], 1.5, 1e-12, "two-element scan out[0]");
        CHECK_CLOSE(out[1], 4.0, 1e-12, "two-element scan out[1]");
    }

    /* ---- Test 4: zero-length edge case ----------------------------- */
    {
        /* Must not crash; nothing to check in out[]. */
        scan(NULL, NULL, 0);
        CHECK(1, "zero-length scan does not crash");
    }

    /* ---- Test 5: oracle checksum at REF_N -------------------------- */
    {
        double cs = run_checksum(REF_N);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle checksum (REF_N)");
    }

    /* ---- Test 6: monotone non-decreasing for positive input -------- */
    {
        long n = 1000;
        double *in  = (double *)malloc(sizeof(double) * (size_t)n);
        double *out = (double *)malloc(sizeof(double) * (size_t)n);
        for (long i = 0; i < n; i++) in[i] = (double)(i + 1);   /* all > 0 */
        scan(in, out, n);
        int mono = 1;
        for (long i = 1; i < n; i++)
            if (out[i] < out[i - 1]) { mono = 0; break; }
        CHECK(mono, "scan is non-decreasing for positive input");
        free(in); free(out);
    }

    return TEST_SUMMARY();
}
