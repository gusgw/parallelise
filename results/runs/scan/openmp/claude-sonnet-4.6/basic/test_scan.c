/* test_scan.c — unit test for the parallel inclusive prefix-sum kernel.
 *
 * Checks:
 *   1. Tiny hand-verifiable case (n=5).
 *   2. Single-element edge case (n=1).
 *   3. Full oracle case (n=REF_N): checksum must match REF_CHECKSUM within REF_TOL.
 *
 * Build:  gcc -O2 -fopenmp -Wall -Wextra -std=c11 test_scan.c scan.o -o test_scan -lm
 * Run:    OMP_NUM_THREADS=1  ./test_scan
 *         OMP_NUM_THREADS=$(nproc) ./test_scan
 */

#include <math.h>     /* fabs  */
#include <stdlib.h>   /* malloc, free */
#include <stdio.h>    /* printf */

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ */
/* Helper: check every element of a tiny scan against expected values. */
/* ------------------------------------------------------------------ */
static void check_elements(const double *out, const double *want, long n,
                            const char *label)
{
    for (long i = 0; i < n; i++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: out[%ld]", label, i);
        CHECK_CLOSE(out[i], want[i], 1e-12, msg);
    }
}

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* Test 1: tiny hand-computed case, n = 5.                             */
    /* in  = {1, 2, 3, 4, 5}                                               */
    /* out = {1, 3, 6, 10, 15}                                             */
    /* ------------------------------------------------------------------ */
    {
        double in5[5]   = {1.0, 2.0, 3.0, 4.0, 5.0};
        double out5[5]  = {0};
        double want5[5] = {1.0, 3.0, 6.0, 10.0, 15.0};
        scan(in5, out5, 5);
        check_elements(out5, want5, 5, "tiny n=5");
    }

    /* ------------------------------------------------------------------ */
    /* Test 2: single-element edge case, n = 1.                            */
    /* ------------------------------------------------------------------ */
    {
        double in1[1]   = {42.0};
        double out1[1]  = {0};
        double want1[1] = {42.0};
        scan(in1, out1, 1);
        check_elements(out1, want1, 1, "single n=1");
    }

    /* ------------------------------------------------------------------ */
    /* Test 3: oracle case, n = REF_N.                                     */
    /* Fill with the deterministic pseudo-random sequence from problem.h,  */
    /* run the kernel, compute the checksum, compare to REF_CHECKSUM.      */
    /* ------------------------------------------------------------------ */
    {
        double *in  = (double *)malloc(sizeof(double) * REF_N);
        double *out = (double *)malloc(sizeof(double) * REF_N);
        if (!in || !out) {
            printf("FAIL: malloc failed for oracle test\n");
            free(in); free(out);
            return 1;
        }

        for (long i = 0; i < REF_N; i++) in[i] = in_of(i);

        scan(in, out, REF_N);

        double cs = checksum(out, REF_N);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle checksum n=REF_N");

        free(in);
        free(out);
    }

    return TEST_SUMMARY();
}
