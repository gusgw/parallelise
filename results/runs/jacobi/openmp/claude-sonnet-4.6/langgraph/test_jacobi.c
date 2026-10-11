/* test_jacobi.c — unit tests for the Jacobi heat-stencil kernel.
 *
 * Checks:
 *   1. Boundary rows/columns are never modified (fixed edges).
 *   2. A single sweep on a tiny grid matches hand-computed values.
 *   3. The full REF_N × REF_N grid after REF_ITERS sweeps reproduces the
 *      notebook checksum within REF_TOL.
 *   4. Zero-iteration call leaves the grid unchanged.
 *   5. Two identical runs produce bit-identical results (determinism).
 */

#include <math.h>     /* fabs   */
#include <stdlib.h>   /* malloc, free */
#include <stdio.h>    /* printf */

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

/* Allocate and fill an n×n grid using the shared u_init helper from problem.h */
static double *make_grid(long n)
{
    double *u = (double *)malloc(sizeof(double) * n * n);
    if (!u) return NULL;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);
    return u;
}

/* ------------------------------------------------------------------ tests   */

/* Test 1: boundaries are fixed after any number of sweeps.
 *
 * Initial conditions (from u_init / problem.h):
 *   top row (i==0)  : 1.0  (hot)
 *   all other edges : 0.0  (cold)
 *
 * After sweeps the edges must still hold those same initial values.
 */
static void test_boundaries_fixed(void)
{
    const long n = 16;
    double *u = make_grid(n);
    if (!u) { CHECK(0, "allocation failed"); return; }

    jacobi(u, n, 50);

    /* Top row must stay 1.0 (hot boundary) */
    for (long j = 0; j < n; j++)
        CHECK(u[0 * n + j] == 1.0, "top boundary must stay 1.0");

    /* Bottom row must stay 0.0 */
    for (long j = 0; j < n; j++)
        CHECK(u[(n-1) * n + j] == 0.0, "bottom boundary must stay 0.0");

    /* Left column: i==0 corner is part of the hot top row (1.0); all other
     * rows in the left column are cold (0.0). */
    CHECK(u[0 * n + 0] == 1.0, "top-left corner must stay 1.0");
    for (long i = 1; i < n; i++)
        CHECK(u[i * n + 0] == 0.0, "left boundary (non-top) must stay 0.0");

    /* Right column: same logic — top-right corner is hot, rest cold. */
    CHECK(u[0 * n + (n-1)] == 1.0, "top-right corner must stay 1.0");
    for (long i = 1; i < n; i++)
        CHECK(u[i * n + (n-1)] == 0.0, "right boundary (non-top) must stay 0.0");

    free(u);
}

/* Test 2: one sweep on a 4×4 grid matches hand-computed stencil values.
 *
 * Initial grid (top row hot = 1.0, rest 0.0):
 *   1  1  1  1
 *   0  0  0  0
 *   0  0  0  0
 *   0  0  0  0
 *
 * After one sweep the only interior cells are (1,1), (1,2), (2,1), (2,2).
 * All stencil reads come from the PREVIOUS (initial) grid:
 *   u_new[1][1] = 0.25*(u[0][1] + u[2][1] + u[1][0] + u[1][2])
 *               = 0.25*(1 + 0 + 0 + 0) = 0.25
 *   u_new[1][2] = 0.25*(u[0][2] + u[2][2] + u[1][1] + u[1][3])
 *               = 0.25*(1 + 0 + 0 + 0) = 0.25
 *   u_new[2][1] = 0.25*(u[1][1] + u[3][1] + u[2][0] + u[2][2])
 *               = 0.25*(0 + 0 + 0 + 0) = 0.0
 *   u_new[2][2] = 0.25*(u[1][2] + u[3][2] + u[2][1] + u[2][3])
 *               = 0.25*(0 + 0 + 0 + 0) = 0.0
 */
static void test_single_sweep_4x4(void)
{
    const long n = 4;
    double *u = make_grid(n);
    if (!u) { CHECK(0, "allocation failed"); return; }

    jacobi(u, n, 1);

    const double tol = 1e-15;
    CHECK_CLOSE(u[1*n+1], 0.25, tol, "4x4 sweep: u[1][1]");
    CHECK_CLOSE(u[1*n+2], 0.25, tol, "4x4 sweep: u[1][2]");
    CHECK_CLOSE(u[2*n+1], 0.00, tol, "4x4 sweep: u[2][1]");
    CHECK_CLOSE(u[2*n+2], 0.00, tol, "4x4 sweep: u[2][2]");

    free(u);
}

/* Test 3: zero sweeps — grid must be unchanged. */
static void test_zero_iters(void)
{
    const long n = 8;
    double *u = make_grid(n);
    if (!u) { CHECK(0, "allocation failed"); return; }

    jacobi(u, n, 0);

    /* Interior cell (1,1) must still be 0.0 */
    CHECK_CLOSE(u[1*n+1], 0.0, 1e-15, "zero iters: interior unchanged");
    /* Top boundary still 1.0 */
    CHECK_CLOSE(u[0*n+1], 1.0, 1e-15, "zero iters: top boundary unchanged");

    free(u);
}

/* Test 4: full oracle check — REF_N × REF_N, REF_ITERS sweeps. */
static void test_oracle_checksum(void)
{
    double *u = make_grid(REF_N);
    if (!u) { CHECK(0, "allocation failed for oracle test"); return; }

    jacobi(u, REF_N, (int)REF_ITERS);

    double cs = checksum(u, REF_N);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle checksum");

    free(u);
}

/* Test 5: result must be independent of thread count (determinism).
 * Run the same problem twice and compare cell-by-cell. */
static void test_determinism(void)
{
    const long n = 32;
    const int  iters = 20;

    double *u1 = make_grid(n);
    double *u2 = make_grid(n);
    if (!u1 || !u2) { CHECK(0, "allocation failed"); free(u1); free(u2); return; }

    jacobi(u1, n, iters);
    jacobi(u2, n, iters);

    int ok = 1;
    for (long k = 0; k < n * n; k++) {
        if (u1[k] != u2[k]) { ok = 0; break; }
    }
    CHECK(ok, "two identical runs must produce bit-identical results");

    free(u1);
    free(u2);
}

/* ------------------------------------------------------------------ main    */

int main(void)
{
    printf("=== test_jacobi ===\n");

    test_boundaries_fixed();
    test_single_sweep_4x4();
    test_zero_iters();
    test_oracle_checksum();
    test_determinism();

    return TEST_SUMMARY();
}
