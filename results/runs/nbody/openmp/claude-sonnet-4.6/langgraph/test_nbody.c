/* test_nbody.c — Unit test for nbody_accel().
 *
 * Checks:
 *   1. The checksum sum_i(|ax_i|+|ay_i|) matches the Python oracle within REF_TOL
 *      (relative tolerance).
 *   2. A tiny hand-verifiable 2-body case: two unit-mass bodies separated by 1 on
 *      the x-axis.  With softening EPS2=0.0025 the expected acceleration magnitude
 *      is 1 / ((1 + 0.0025) * sqrt(1 + 0.0025)).
 *   3. The kernel is deterministic: two back-to-back calls on the same input give
 *      bit-identical results.
 */

#include <math.h>    /* fabs, sqrt */
#include <stdlib.h>  /* malloc, free */
#include <string.h>  /* memcmp */
#include <stdio.h>   /* printf (via testing.h) */

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

static double *alloc_vec(long n)
{
    double *v = malloc(sizeof(double) * (size_t)n);
    if (!v) { fprintf(stderr, "malloc failed\n"); exit(1); }
    return v;
}

/* ------------------------------------------------------------------ tests   */

/* Test 1: oracle checksum at N = REF_N */
static void test_oracle_checksum(void)
{
    long n = REF_N;
    double *px = alloc_vec(n);
    double *py = alloc_vec(n);
    double *ax = alloc_vec(n);
    double *ay = alloc_vec(n);

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    double cs = checksum(ax, ay, n);

    /* relative tolerance: |got - want| / |want| <= REF_TOL */
    double rel_err = fabs(cs - REF_CHECKSUM) / fabs(REF_CHECKSUM);
    CHECK_CLOSE(rel_err, 0.0, REF_TOL,
                "oracle checksum (relative error vs REF_CHECKSUM)");

    free(px); free(py); free(ax); free(ay);
}

/* Test 2: 2-body analytic case */
static void test_two_body(void)
{
    /* Body 0 at x=0, body 1 at x=1; both at y=0.
     * dx = 1, dy = 0, r2 = 1.
     * d3 = (1 + EPS2) * sqrt(1 + EPS2)
     * ax[0] = +1/d3,  ax[1] = -1/d3,  ay[0] = ay[1] = 0.
     */
    double eps2 = 0.0025;
    double r2s  = 1.0 + eps2;
    double d3   = r2s * sqrt(r2s);
    double expected = 1.0 / d3;

    double px[2] = {0.0, 1.0};
    double py[2] = {0.0, 0.0};
    double ax[2], ay[2];

    nbody_accel(px, py, ax, ay, 2L);

    double tol = 1e-14;   /* should be exact to floating-point precision */
    CHECK_CLOSE(ax[0],  expected, tol, "2-body: ax[0]");
    CHECK_CLOSE(ax[1], -expected, tol, "2-body: ax[1]");
    CHECK_CLOSE(ay[0],  0.0,      tol, "2-body: ay[0]");
    CHECK_CLOSE(ay[1],  0.0,      tol, "2-body: ay[1]");
}

/* Test 3: determinism — two calls produce bit-identical output */
static void test_determinism(void)
{
    long n = 256;
    double *px = alloc_vec(n);
    double *py = alloc_vec(n);
    double *ax1 = alloc_vec(n);
    double *ay1 = alloc_vec(n);
    double *ax2 = alloc_vec(n);
    double *ay2 = alloc_vec(n);

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax1, ay1, n);
    nbody_accel(px, py, ax2, ay2, n);

    int same = (memcmp(ax1, ax2, sizeof(double) * (size_t)n) == 0 &&
                memcmp(ay1, ay2, sizeof(double) * (size_t)n) == 0);
    CHECK(same, "determinism: two calls give bit-identical ax/ay");

    free(px); free(py); free(ax1); free(ay1); free(ax2); free(ay2);
}

/* Test 4: single-body edge case — no interactions, accelerations must be 0 */
static void test_single_body(void)
{
    double px[1] = {0.5};
    double py[1] = {-0.3};
    double ax[1], ay[1];

    nbody_accel(px, py, ax, ay, 1L);

    CHECK_CLOSE(ax[0], 0.0, 1e-30, "single body: ax[0] == 0");
    CHECK_CLOSE(ay[0], 0.0, 1e-30, "single body: ay[0] == 0");
}

/* ------------------------------------------------------------------ main    */

int main(void)
{
    test_oracle_checksum();
    test_two_body();
    test_determinism();
    test_single_body();

    return TEST_SUMMARY();
}
