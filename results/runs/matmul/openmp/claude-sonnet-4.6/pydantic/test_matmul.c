/* test_matmul.c — Unit test for the matmul kernel.
 *
 * Fills A and B with the same deterministic integer pattern used in the Python
 * notebook (via problem.h), runs matmul(), computes the checksum of C, and
 * checks it against the oracle value from reference.h within REF_TOL.
 *
 * Additional sanity checks:
 *   - a tiny 2×2 hand-computed case
 *   - the full REF_N × REF_N checksum
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

static double *alloc_matrix(long n)
{
    double *m = malloc(sizeof(double) * (size_t)(n * n));
    if (!m) { fprintf(stderr, "alloc_matrix: out of memory\n"); exit(1); }
    return m;
}

static void fill_A(double *A, long n)
{
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            A[i * n + j] = A_of(i, j);
}

static void fill_B(double *B, long n)
{
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            B[i * n + j] = B_of(i, j);
}

/* ------------------------------------------------------------------ tests */

/* 2×2 hand-computed sanity check.
 *
 *   A = [[A_of(0,0), A_of(0,1)],   B = [[B_of(0,0), B_of(0,1)],
 *        [A_of(1,0), A_of(1,1)]]        [B_of(1,0), B_of(1,1)]]
 *
 * A_of(i,j) = ((i*3 + j*7) % 11) + 1
 *   A[0][0]=1, A[0][1]=8, A[1][0]=4, A[1][1]=11
 *
 * B_of(i,j) = ((i*5 + j*2) % 13) + 1
 *   B[0][0]=1, B[0][1]=3, B[1][0]=6, B[1][1]=8
 *
 * C[0][0] = 1*1 + 8*6 = 49
 * C[0][1] = 1*3 + 8*8 = 67
 * C[1][0] = 4*1 + 11*6 = 70
 * C[1][1] = 4*3 + 11*8 = 100
 */
static void test_2x2(void)
{
    double A[4], B[4], C[4];
    fill_A(A, 2);
    fill_B(B, 2);
    matmul(A, B, C, 2);

    CHECK_CLOSE(C[0 * 2 + 0], 49.0,  REF_TOL, "2x2 C[0][0]");
    CHECK_CLOSE(C[0 * 2 + 1], 67.0,  REF_TOL, "2x2 C[0][1]");
    CHECK_CLOSE(C[1 * 2 + 0], 70.0,  REF_TOL, "2x2 C[1][0]");
    CHECK_CLOSE(C[1 * 2 + 1], 100.0, REF_TOL, "2x2 C[1][1]");
}

/* Full REF_N × REF_N checksum must match the Python oracle. */
static void test_checksum(void)
{
    long n = REF_N;
    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);

    fill_A(A, n);
    fill_B(B, n);
    matmul(A, B, C, n);

    double cs = checksum(C, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "full checksum vs oracle");

    free(A);
    free(B);
    free(C);
}

/* Result must be independent of thread count (idempotency check: run twice). */
static void test_idempotent(void)
{
    long n = 64;
    double *A  = alloc_matrix(n);
    double *B  = alloc_matrix(n);
    double *C1 = alloc_matrix(n);
    double *C2 = alloc_matrix(n);

    fill_A(A, n);
    fill_B(B, n);
    matmul(A, B, C1, n);
    matmul(A, B, C2, n);

    double cs1 = checksum(C1, n);
    double cs2 = checksum(C2, n);
    CHECK_CLOSE(cs1, cs2, REF_TOL, "idempotent: two runs give same checksum");

    free(A); free(B); free(C1); free(C2);
}

/* ------------------------------------------------------------------ main */

int main(void)
{
    test_2x2();
    test_checksum();
    test_idempotent();
    return TEST_SUMMARY();
}
