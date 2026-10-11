/* test_matmul.c — Unit tests for the matmul kernel.
 *
 * Tests performed:
 *   1. Tiny 2×2 known-answer check (exact integer arithmetic).
 *   2. Tiny 3×3 known-answer check (exact integer arithmetic).
 *   3. REF_N × REF_N checksum check against the Python oracle (REF_CHECKSUM),
 *      using the deterministic fill from problem.h.  Must be within REF_TOL.
 *   4. Thread-count independence: run the full-size kernel a second time with
 *      a different OMP_NUM_THREADS setting and verify the checksum is identical.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* ------------------------------------------------------------------ helpers */

/* Allocate and fill an n×n matrix using the supplied fill function. */
static double *make_matrix(long n, double (*fill)(long, long))
{
    double *M = malloc(sizeof(double) * (size_t)(n * n));
    if (!M) { fprintf(stderr, "malloc failed\n"); exit(1); }
    if (fill) {
        for (long i = 0; i < n; i++)
            for (long j = 0; j < n; j++)
                M[i * n + j] = fill(i, j);
    } else {
        for (long t = 0; t < n * n; t++) M[t] = 0.0;
    }
    return M;
}

/* ------------------------------------------------------------------ tests */

/* Test 1 & 2: small known-answer checks. */
static void test_tiny(void)
{
    /* 2×2:
     *   A = [[1,2],[3,4]]   B = [[5,6],[7,8]]
     *   C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
     *     = [[19,22],[43,50]]
     */
    {
        double A[4] = {1, 2, 3, 4};
        double B[4] = {5, 6, 7, 8};
        double C[4] = {0};
        matmul(A, B, C, 2);
        CHECK_CLOSE(C[0], 19.0, 1e-10, "2x2 C[0,0]");
        CHECK_CLOSE(C[1], 22.0, 1e-10, "2x2 C[0,1]");
        CHECK_CLOSE(C[2], 43.0, 1e-10, "2x2 C[1,0]");
        CHECK_CLOSE(C[3], 50.0, 1e-10, "2x2 C[1,1]");
    }

    /* 3×3:
     *   A = I (identity)   =>   C = B
     */
    {
        double A[9] = {1,0,0, 0,1,0, 0,0,1};
        double B[9] = {1,2,3, 4,5,6, 7,8,9};
        double C[9] = {0};
        matmul(A, B, C, 3);
        for (int t = 0; t < 9; t++)
            CHECK_CLOSE(C[t], B[t], 1e-10, "3x3 identity: C == B");
    }
}

/* Test 3: full-size checksum against the Python oracle. */
static void test_ref_checksum(void)
{
    long n = REF_N;
    double *A = make_matrix(n, A_of);
    double *B = make_matrix(n, B_of);
    double *C = make_matrix(n, NULL);

    matmul(A, B, C, n);

    double cs = checksum(C, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "REF_N checksum vs Python oracle");

    free(A); free(B); free(C);
}

/* Test 4: result must be independent of thread count.
 * We run the kernel twice (the OMP_NUM_THREADS env var controls the first run;
 * we explicitly set 1 thread for the second run) and compare checksums. */
static void test_thread_independence(void)
{
    long n = REF_N;
    double *A = make_matrix(n, A_of);
    double *B = make_matrix(n, B_of);
    double *C1 = make_matrix(n, NULL);
    double *C2 = make_matrix(n, NULL);

    /* First run: whatever OMP_NUM_THREADS is set to externally. */
    matmul(A, B, C1, n);
    double cs1 = checksum(C1, n);

    /* Second run: force 1 thread. */
    omp_set_num_threads(1);
    matmul(A, B, C2, n);
    double cs2 = checksum(C2, n);

    CHECK_CLOSE(cs1, cs2, REF_TOL,
                "checksum identical regardless of thread count");

    free(A); free(B); free(C1); free(C2);
}

/* ------------------------------------------------------------------ main */

int main(void)
{
    test_tiny();
    test_ref_checksum();
    test_thread_independence();
    return TEST_SUMMARY();
}
