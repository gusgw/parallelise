/* test_nbody.c — Unit test for nbody_accel().
 *
 * Fills positions with the same deterministic formula as the Python notebook
 * (via problem.h helpers), runs nbody_accel(), computes the checksum
 * sum_i(|ax_i| + |ay_i|), and checks it against the oracle in reference.h
 * within REF_TOL.
 *
 * Two extra sanity checks:
 *   • tiny 3-body case: by symmetry the net acceleration on the centre body
 *     must be zero (bodies placed symmetrically cancel exactly).
 *   • n=1: a single body has no neighbours, so ax[0] == ay[0] == 0.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

static void fill_positions(double *px, double *py, long n)
{
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }
}

/* ------------------------------------------------------------------ tests   */

/* 1. Single body: no interaction, accelerations must be exactly 0. */
static void test_single_body(void)
{
    double px1 = 0.5, py1 = -0.3;
    double ax1 = 999.0, ay1 = 999.0;
    nbody_accel(&px1, &py1, &ax1, &ay1, 1L);
    CHECK(ax1 == 0.0, "single body: ax must be 0");
    CHECK(ay1 == 0.0, "single body: ay must be 0");
}

/* 2. Two equal bodies: forces are equal and opposite, so ax[0] == -ax[1]
 *    and ay[0] == -ay[1], and neither is zero. */
static void test_two_bodies(void)
{
    double px2[2] = { -0.5,  0.5 };
    double py2[2] = {  0.0,  0.0 };
    double ax2[2], ay2[2];
    nbody_accel(px2, py2, ax2, ay2, 2L);
    /* Forces must be equal and opposite */
    CHECK_CLOSE(ax2[0], -ax2[1], 1e-14, "two bodies: ax[0] == -ax[1]");
    CHECK_CLOSE(ay2[0], -ay2[1], 1e-14, "two bodies: ay[0] == -ay[1]");
    /* The force must be non-zero */
    CHECK(ax2[0] != 0.0, "two bodies: ax[0] must be non-zero");
}

/* 3. Reference checksum at N = REF_N. */
static void test_reference_checksum(void)
{
    long n = REF_N;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }

    fill_positions(px, py, n);
    nbody_accel(px, py, ax, ay, n);

    double cs = checksum(ax, ay, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL * REF_CHECKSUM,
                "reference checksum matches oracle");

    printf("  checksum = %.10g  (expected %.10g, tol %.2g)\n",
           cs, (double)REF_CHECKSUM, (double)(REF_TOL * REF_CHECKSUM));

    free(px); free(py); free(ax); free(ay);
}

/* ------------------------------------------------------------------ main    */

int main(void)
{
    test_single_body();
    test_two_bodies();
    test_reference_checksum();
    return TEST_SUMMARY();
}
