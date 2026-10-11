/* test_scan.c — unit tests for the parallel inclusive prefix-sum kernel.
 *
 * Tests:
 *   1. Tiny hand-verifiable case (n=5, known exact values).
 *   2. Single-element edge case (n=1).
 *   3. Full oracle case (n=REF_N): checksum must match REF_CHECKSUM within REF_TOL.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ */
/* Test 1: tiny hand-crafted array                                     */
/* ------------------------------------------------------------------ */
static void test_tiny(void)
{
    /* in = {1, 2, 3, 4, 5}  →  out = {1, 3, 6, 10, 15} */
    const double in[5]       = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double expected[5] = {1.0, 3.0, 6.0, 10.0, 15.0};
    double out[5]            = {0};

    scan(in, out, 5);

    for (int i = 0; i < 5; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "tiny: out[%d]", i);
        CHECK_CLOSE(out[i], expected[i], 1e-12, msg);
    }
}

/* ------------------------------------------------------------------ */
/* Test 2: single-element array                                        */
/* ------------------------------------------------------------------ */
static void test_single(void)
{
    const double in[1]  = {42.0};
    double       out[1] = {0.0};

    scan(in, out, 1);
    CHECK_CLOSE(out[0], 42.0, 1e-12, "single element");
}

/* ------------------------------------------------------------------ */
/* Test 3: oracle checksum for n = REF_N                               */
/* ------------------------------------------------------------------ */
static void test_oracle(void)
{
    long n = REF_N;

    double *in  = (double *)malloc(sizeof(double) * (size_t)n);
    double *out = (double *)malloc(sizeof(double) * (size_t)n);
    if (!in || !out) {
        fprintf(stderr, "malloc failed in test_oracle\n");
        free(in); free(out);
        return;
    }

    /* Deterministic fill (matches the Python notebook exactly). */
    for (long i = 0; i < n; i++) in[i] = in_of(i);

    scan(in, out, n);

    double cs = checksum(out, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle checksum");

    free(in);
    free(out);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    test_tiny();
    test_single();
    test_oracle();
    return TEST_SUMMARY();
}
