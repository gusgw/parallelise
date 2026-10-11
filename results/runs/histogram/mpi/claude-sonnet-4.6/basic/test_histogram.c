/* test_histogram.c — unit test for the parallel histogram kernel.
 *
 * Every MPI rank calls histogram() the same number of times (required so that
 * the collective inside the kernel does not deadlock).  Only rank 0 checks the
 * results against reference.h and prints the test summary.
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ------------------------------------------------------------------ */
    /* Build the input array (every rank holds the full array, as required */
    /* by the histogram.h contract).                                       */
    /* ------------------------------------------------------------------ */
    long n     = REF_N;
    int  nbins = (int)REF_NBINS;

    double *x      = malloc(sizeof(double) * (size_t)n);
    long   *counts = malloc(sizeof(long)   * (size_t)nbins);
    if (!x || !counts) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) x[i] = x_of(i);

    /* ------------------------------------------------------------------ */
    /* Call the collective kernel (all ranks participate).                 */
    /* ------------------------------------------------------------------ */
    histogram(x, n, counts, nbins);

    /* ------------------------------------------------------------------ */
    /* Validate on rank 0 only.                                            */
    /* ------------------------------------------------------------------ */
    if (rank == 0) {
        /* 1. Total count must equal n */
        long total = 0;
        for (int b = 0; b < nbins; b++) total += counts[b];
        CHECK(total == n, "total count == n");

        /* 2. Weighted checksum must match the oracle within REF_TOL */
        double cs = checksum(counts, nbins);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches oracle");

        /* 3. No bin may have a negative count */
        int any_negative = 0;
        for (int b = 0; b < nbins; b++) {
            if (counts[b] < 0) { any_negative = 1; break; }
        }
        CHECK(!any_negative, "all bin counts non-negative");
    }

    free(x);
    free(counts);

    int result = 0;
    if (rank == 0) result = TEST_SUMMARY();

    MPI_Finalize();
    return result;
}
