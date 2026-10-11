/* test_jacobi.c — unit tests for the Jacobi heat stencil kernel.
 *
 * Tests:
 *   1. Full oracle run: REF_N x REF_N grid, REF_ITERS sweeps.
 *      Checksum must match REF_CHECKSUM within REF_TOL.
 *   2. Boundary preservation: after any number of sweeps the four edges must
 *      remain exactly at their initial values (top row = 1.0, others = 0.0).
 *   3. Small-grid sanity: a 4x4 grid, 1 sweep — hand-verify one interior cell.
 *   4. Zero-iteration identity: calling jacobi with iters=0 must leave the
 *      grid unchanged.
 *   5. Determinism: result is identical for 1 thread and max threads.
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ */
/* Helper: allocate and fill an n x n grid using u_init from problem.h */
/* ------------------------------------------------------------------ */
static double *make_grid(long n)
{
    double *u = (double *)malloc(sizeof(double) * n * n);
    if (!u) return NULL;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);
    return u;
}

/* ------------------------------------------------------------------ */
/* Test 1: oracle checksum                                             */
/* ------------------------------------------------------------------ */
static void test_oracle_checksum(void)
{
    double *u = make_grid(REF_N);
    if (!u) { CHECK(0, "oracle: allocation failed"); return; }

    jacobi(u, REF_N, (int)REF_ITERS);

    double cs = checksum(u, REF_N);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "oracle checksum");

    free(u);
}

/* ------------------------------------------------------------------ */
/* Test 2: boundary rows/columns are never modified                    */
/*                                                                     */
/* Initial conditions (from u_init / problem.h):                      */
/*   top row (i==0): 1.0 for all j                                     */
/*   all other cells: 0.0                                              */
/* So the expected boundary values after any number of sweeps are:    */
/*   top row    -> 1.0                                                 */
/*   bottom row -> 0.0                                                 */
/*   left col   -> 1.0 at i==0 (corner), 0.0 elsewhere                */
/*   right col  -> 1.0 at i==0 (corner), 0.0 elsewhere                */
/* ------------------------------------------------------------------ */
static void test_boundaries_preserved(void)
{
    long n = 32;
    double *u = make_grid(n);
    if (!u) { CHECK(0, "boundary: allocation failed"); return; }

    jacobi(u, n, 50);

    int ok = 1;

    /* Top row must stay 1.0 */
    for (long j = 0; j < n; j++)
        if (u[0 * n + j] != 1.0) { ok = 0; break; }

    /* Bottom row must stay 0.0 */
    for (long j = 0; j < n && ok; j++)
        if (u[(n - 1) * n + j] != 0.0) { ok = 0; break; }

    /* Left column: corner (i==0) is 1.0, rest are 0.0 */
    for (long i = 1; i < n && ok; i++)
        if (u[i * n + 0] != 0.0) { ok = 0; break; }

    /* Right column: corner (i==0) is 1.0, rest are 0.0 */
    for (long i = 1; i < n && ok; i++)
        if (u[i * n + (n - 1)] != 0.0) { ok = 0; break; }

    CHECK(ok, "boundary cells unchanged after 50 sweeps");
    free(u);
}

/* ------------------------------------------------------------------ */
/* Test 3: single sweep on a small grid — hand-computed cell value     */
/*                                                                     */
/* 4x4 grid (row-major), top row hot (1.0), rest 0.0:                 */
/*   row 0: 1 1 1 1   (boundary, fixed)                               */
/*   row 1: 0 0 0 0                                                    */
/*   row 2: 0 0 0 0                                                    */
/*   row 3: 0 0 0 0   (boundary, fixed)                               */
/*                                                                     */
/* After 1 sweep, interior cells (i in {1,2}, j in {1,2}):            */
/*   u[1][1] = 0.25*(u[0][1]+u[2][1]+u[1][0]+u[1][2])                */
/*           = 0.25*(1 + 0 + 0 + 0) = 0.25                            */
/*   u[1][2] = 0.25*(u[0][2]+u[2][2]+u[1][1]+u[1][3])                */
/*           = 0.25*(1 + 0 + 0 + 0) = 0.25                            */
/* (u[1][1] and u[1][2] both read from the OLD grid, so both = 0.25)  */
/* ------------------------------------------------------------------ */
static void test_single_sweep_small(void)
{
    long n = 4;
    double *u = make_grid(n);
    if (!u) { CHECK(0, "small: allocation failed"); return; }

    jacobi(u, n, 1);

    /* Interior cells in row 1 each have exactly one hot neighbour (row 0). */
    CHECK_CLOSE(u[1 * n + 1], 0.25, 1e-15, "small grid: u[1][1] after 1 sweep");
    CHECK_CLOSE(u[1 * n + 2], 0.25, 1e-15, "small grid: u[1][2] after 1 sweep");

    /* Interior cells in row 2 have no hot neighbours (row 1 was still 0 in
     * the old grid), so they stay 0.0. */
    CHECK_CLOSE(u[2 * n + 1], 0.0, 1e-15, "small grid: u[2][1] after 1 sweep");
    CHECK_CLOSE(u[2 * n + 2], 0.0, 1e-15, "small grid: u[2][2] after 1 sweep");

    free(u);
}

/* ------------------------------------------------------------------ */
/* Test 4: zero iterations — grid must be identical to initial state   */
/* ------------------------------------------------------------------ */
static void test_zero_iters_identity(void)
{
    long n = 16;
    double *u    = make_grid(n);
    double *orig = make_grid(n);
    if (!u || !orig) {
        CHECK(0, "zero-iters: allocation failed");
        free(u); free(orig);
        return;
    }

    jacobi(u, n, 0);

    int ok = 1;
    for (long t = 0; t < n * n; t++)
        if (u[t] != orig[t]) { ok = 0; break; }

    CHECK(ok, "zero iters: grid unchanged");
    free(u);
    free(orig);
}

/* ------------------------------------------------------------------ */
/* Test 5: result is independent of thread count (determinism)         */
/*                                                                     */
/* Run with 1 thread and with the default thread count; checksums must */
/* match exactly (the stencil is deterministic for any thread count).  */
/* ------------------------------------------------------------------ */
static void test_determinism(void)
{
    long n = 64;
    int  iters = 30;

    double *u1 = make_grid(n);
    double *u2 = make_grid(n);
    if (!u1 || !u2) {
        CHECK(0, "determinism: allocation failed");
        free(u1); free(u2);
        return;
    }

    /* Run with 1 thread */
    omp_set_num_threads(1);
    jacobi(u1, n, iters);
    double cs1 = checksum(u1, n);

    /* Run with default (possibly many) threads — caller sets OMP_NUM_THREADS */
    omp_set_num_threads(omp_get_max_threads());
    jacobi(u2, n, iters);
    double cs2 = checksum(u2, n);

    CHECK_CLOSE(cs1, cs2, 1e-10, "determinism: 1-thread vs max-thread checksum");

    free(u1);
    free(u2);
}

/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=== jacobi unit tests ===\n");
    test_oracle_checksum();
    test_boundaries_preserved();
    test_single_sweep_small();
    test_zero_iters_identity();
    test_determinism();
    return TEST_SUMMARY();
}
