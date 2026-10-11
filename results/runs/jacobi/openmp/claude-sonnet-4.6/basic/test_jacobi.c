/* test_jacobi.c — unit tests for the Jacobi heat-stencil kernel.
 *
 * Checks:
 *   1. Single-sweep sanity: a 4x4 grid with the top row hot, one sweep, verify one
 *      interior cell by hand.
 *   2. Reference oracle: REF_N x REF_N grid, REF_ITERS sweeps, checksum must match
 *      REF_CHECKSUM within REF_TOL (values from reference.h / jacobi.ipynb).
 *   3. Boundary invariance: after any number of sweeps the four edges must be unchanged.
 *   4. Idempotence of zero iterations: calling jacobi with iters==0 must leave the
 *      grid untouched.
 *   5. Odd vs even iteration count: result must be in the caller's buffer either way.
 */

#include <math.h>     /* fabs   */
#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memcpy */
#include <stdio.h>    /* printf */

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

/* Allocate and fill an n x n grid using the shared u_init helper from problem.h. */
static double *make_grid(long n)
{
    double *u = (double *)malloc(sizeof(double) * n * n);
    if (!u) return NULL;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);
    return u;
}

/* ------------------------------------------------------------------ tests */

/* Test 1: hand-computed single sweep on a small 4x4 grid.
 *
 * Initial grid (top row hot = 1.0, rest 0.0):
 *   1  1  1  1
 *   0  0  0  0
 *   0  0  0  0
 *   0  0  0  0
 *
 * After 1 sweep the only interior cells are (1,1),(1,2),(2,1),(2,2).
 * All four neighbours of (1,1) from the PREVIOUS grid:
 *   u_new[1][1] = 0.25*(u[0][1] + u[2][1] + u[1][0] + u[1][2])
 *               = 0.25*(1.0     + 0.0     + 0.0     + 0.0    ) = 0.25
 *   u_new[1][2] = 0.25*(u[0][2] + u[2][2] + u[1][1] + u[1][3])
 *               = 0.25*(1.0     + 0.0     + 0.0     + 0.0    ) = 0.25
 *   u_new[2][1] = 0.25*(u[1][1] + u[3][1] + u[2][0] + u[2][2])
 *               = 0.25*(0.0     + 0.0     + 0.0     + 0.0    ) = 0.0
 *   u_new[2][2] = 0.25*(u[1][2] + u[3][2] + u[2][1] + u[2][3])
 *               = 0.25*(0.0     + 0.0     + 0.0     + 0.0    ) = 0.0
 */
static void test_single_sweep_small(void)
{
    const long n = 4;
    double *u = make_grid(n);

    jacobi(u, n, 1);

    CHECK_CLOSE(u[1 * n + 1], 0.25, 1e-12, "4x4 sweep: u[1][1]");
    CHECK_CLOSE(u[1 * n + 2], 0.25, 1e-12, "4x4 sweep: u[1][2]");
    CHECK_CLOSE(u[2 * n + 1], 0.0,  1e-12, "4x4 sweep: u[2][1]");
    CHECK_CLOSE(u[2 * n + 2], 0.0,  1e-12, "4x4 sweep: u[2][2]");

    free(u);
}

/* Test 2: oracle checksum — the main correctness gate. */
static void test_oracle_checksum(void)
{
    double *u = make_grid(REF_N);

    jacobi(u, REF_N, (int)REF_ITERS);

    double cs = checksum(u, REF_N);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "oracle checksum (REF_N x REF_N, REF_ITERS sweeps)");

    free(u);
}

/* Test 3: boundary rows and columns must never change.
 *
 * The initial condition (from problem.h / jacobi.ipynb):
 *   top row    (i==0): all 1.0  (hot boundary)
 *   bottom row (i==n-1): all 0.0
 *   left col   (j==0): 1.0 at corner (i==0), 0.0 elsewhere
 *   right col  (j==n-1): 1.0 at corner (i==0), 0.0 elsewhere
 *
 * After any number of sweeps the stencil must leave all boundary cells unchanged.
 */
static void test_boundaries_fixed(void)
{
    const long n = 32;
    double *u = make_grid(n);

    /* Save the original boundary values so we can compare exactly. */
    double *orig = (double *)malloc(sizeof(double) * n * n);
    memcpy(orig, u, sizeof(double) * n * n);

    jacobi(u, n, 50);

    /* Top row */
    int top_ok = 1;
    for (long j = 0; j < n; j++)
        if (fabs(u[0 * n + j] - orig[0 * n + j]) > 1e-12) { top_ok = 0; break; }
    CHECK(top_ok, "top boundary row unchanged after 50 sweeps");

    /* Bottom row */
    int bot_ok = 1;
    for (long j = 0; j < n; j++)
        if (fabs(u[(n - 1) * n + j] - orig[(n - 1) * n + j]) > 1e-12) { bot_ok = 0; break; }
    CHECK(bot_ok, "bottom boundary row unchanged after 50 sweeps");

    /* Left column (all rows, including corners) */
    int left_ok = 1;
    for (long i = 0; i < n; i++)
        if (fabs(u[i * n + 0] - orig[i * n + 0]) > 1e-12) { left_ok = 0; break; }
    CHECK(left_ok, "left boundary column unchanged after 50 sweeps");

    /* Right column (all rows, including corners) */
    int right_ok = 1;
    for (long i = 0; i < n; i++)
        if (fabs(u[i * n + (n - 1)] - orig[i * n + (n - 1)]) > 1e-12) { right_ok = 0; break; }
    CHECK(right_ok, "right boundary column unchanged after 50 sweeps");

    free(orig);
    free(u);
}

/* Test 4: zero iterations must leave the grid completely unchanged. */
static void test_zero_iters(void)
{
    const long n = 16;
    double *u    = make_grid(n);
    double *orig = (double *)malloc(sizeof(double) * n * n);
    memcpy(orig, u, sizeof(double) * n * n);

    jacobi(u, n, 0);

    int unchanged = 1;
    for (long t = 0; t < n * n; t++)
        if (fabs(u[t] - orig[t]) > 1e-15) { unchanged = 0; break; }
    CHECK(unchanged, "zero iterations: grid is unchanged");

    free(u);
    free(orig);
}

/* Test 5: odd vs even iteration count — result must be in the caller's buffer either
 * way (exercises the double-buffer swap-back path). */
static void test_odd_even_iters(void)
{
    const long n = 16;

    /* Run 3 sweeps (odd). */
    double *u3 = make_grid(n);
    jacobi(u3, n, 3);
    double cs3 = checksum(u3, n);

    /* Run 4 sweeps (even). */
    double *u4 = make_grid(n);
    jacobi(u4, n, 4);
    double cs4 = checksum(u4, n);

    /* The checksums must differ (more sweeps → more heat diffused). */
    CHECK(fabs(cs4 - cs3) > 1e-6, "odd/even sweep counts give different checksums");

    /* And both must be positive (heat has diffused from the top row). */
    CHECK(cs3 > 0.0, "checksum after 3 sweeps is positive");
    CHECK(cs4 > 0.0, "checksum after 4 sweeps is positive");

    free(u3);
    free(u4);
}

/* ------------------------------------------------------------------ main */

int main(void)
{
    printf("=== jacobi unit tests ===\n");

    test_single_sweep_small();
    test_oracle_checksum();
    test_boundaries_fixed();
    test_zero_iters();
    test_odd_even_iters();

    return TEST_SUMMARY();
}
