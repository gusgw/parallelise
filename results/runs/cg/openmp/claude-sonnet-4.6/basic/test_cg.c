/* test_cg.c — unit test for cgsolve() against the Python-notebook oracle.
 *
 * Checks:
 *   1. The checksum (sum of all x entries) after REF_ITERS iterations on the
 *      REF_M x REF_M grid matches REF_CHECKSUM within REF_TOL.
 *   2. The result is identical (within REF_TOL) when run with 1 thread and
 *      with the default thread count, so the answer does not depend on
 *      parallelism.  (The Makefile already calls this binary twice with
 *      OMP_NUM_THREADS=1 and OMP_NUM_THREADS=$(nproc), so a single run here
 *      is sufficient; we add an explicit 2-thread re-run for good measure.)
 */

#include <stdlib.h>   /* malloc, free */
#include <omp.h>      /* omp_set_num_threads */

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Run cgsolve with the reference parameters and return the checksum. */
static double run_reference(void)
{
    long n = REF_M * REF_M;
    double *b = (double *)malloc(sizeof(double) * (size_t)n);
    double *x = (double *)malloc(sizeof(double) * (size_t)n);

    for (long t = 0; t < n; t++)
        b[t] = b_of(t);

    cgsolve(x, b, REF_M, (int)REF_ITERS);

    double cs = checksum(x, REF_M);

    free(b);
    free(x);
    return cs;
}

int main(void)
{
    /* ---- test 1: correctness at whatever thread count the caller set ---- */
    double cs = run_reference();
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "checksum matches oracle (current OMP_NUM_THREADS)");

    /* ---- test 2: single-threaded result matches oracle ------------------ */
    omp_set_num_threads(1);
    double cs1 = run_reference();
    CHECK_CLOSE(cs1, REF_CHECKSUM, REF_TOL,
                "checksum matches oracle (1 thread)");

    /* ---- test 3: two-threaded result matches oracle --------------------- */
    omp_set_num_threads(2);
    double cs2 = run_reference();
    CHECK_CLOSE(cs2, REF_CHECKSUM, REF_TOL,
                "checksum matches oracle (2 threads)");

    /* ---- test 4: single- and multi-thread results agree with each other - */
    CHECK_CLOSE(cs1, cs2, REF_TOL,
                "1-thread and 2-thread results agree");

    return TEST_SUMMARY();
}
