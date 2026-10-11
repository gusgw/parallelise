/* test_histogram.c -- unit test for the parallel histogram kernel.
 *
 * Every rank calls histogram() the same number of times (required so that the
 * collective inside the kernel does not deadlock).  Only rank 0 checks the
 * result against the oracle in reference.h and prints a pass/fail message.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>     /* fabs */

#include "histogram.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Build the full input array (every rank holds a copy, as specified). */
    long   n     = REF_N;
    int    nbins = (int)REF_NBINS;

    double *x      = malloc(sizeof(double) * (size_t)n);
    long   *counts = malloc(sizeof(long)   * (size_t)nbins);
    if (!x || !counts) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) x[i] = x_of(i);

    /* --- collective call (all ranks participate) --- */
    histogram(x, n, counts, nbins);

    /* --- verification on rank 0 only --- */
    if (rank == 0) {
        /* 1. Total count must equal n. */
        long total = 0;
        for (int b = 0; b < nbins; b++) total += counts[b];

        /* 2. Weighted checksum must match the oracle within REF_TOL. */
        double cs   = checksum(counts, nbins);
        double diff = fabs(cs - REF_CHECKSUM);

        int ok = (total == n) && (diff <= REF_TOL);

        printf("total    = %ld  (expected %ld)  %s\n",
               total, n, total == n ? "OK" : "FAIL");
        printf("checksum = %.1f  (expected %.1f)  diff = %.6f  tol = %.1f  %s\n",
               cs, REF_CHECKSUM, diff, REF_TOL, diff <= REF_TOL ? "OK" : "FAIL");
        printf("%s\n", ok ? "PASS" : "FAIL");

        free(x);
        free(counts);
        MPI_Finalize();
        return ok ? 0 : 1;
    }

    free(x);
    free(counts);
    MPI_Finalize();
    return 0;
}
