/* test_histogram.c – unit test for the parallel histogram kernel
 *
 * Every rank calls histogram() the same number of times (required so that the
 * collective inside the kernel does not deadlock).  Only rank 0 checks the
 * result against the oracle in reference.h and prints a pass/fail message.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 test_histogram.c histogram.o -o test_histogram -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_histogram
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

    /* ------------------------------------------------------------------ */
    /* Build the deterministic input (every rank holds the full array)     */
    /* ------------------------------------------------------------------ */
    const long   n     = REF_N;
    const int    nbins = (int)REF_NBINS;

    double *x      = (double *)malloc(sizeof(double) * n);
    long   *counts = (long   *)malloc(sizeof(long)   * nbins);

    if (!x || !counts) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) x[i] = x_of(i);

    /* ------------------------------------------------------------------ */
    /* Call the collective kernel (all ranks participate)                  */
    /* ------------------------------------------------------------------ */
    histogram(x, n, counts, nbins);

    /* ------------------------------------------------------------------ */
    /* Verify on rank 0 only                                               */
    /* ------------------------------------------------------------------ */
    if (rank == 0) {
        /* 1. Total count must equal n */
        long total = 0;
        for (int b = 0; b < nbins; b++) total += counts[b];

        if (total != n) {
            fprintf(stderr, "FAIL: total count = %ld, expected %ld\n", total, n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* 2. Weighted checksum must match the oracle within REF_TOL */
        double cs  = checksum(counts, nbins);
        double err = fabs(cs - REF_CHECKSUM);

        if (err > REF_TOL) {
            fprintf(stderr,
                    "FAIL: checksum = %.1f, expected %.1f, |err| = %.6f > tol %.6f\n",
                    cs, REF_CHECKSUM, err, (double)REF_TOL);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("PASS: total = %ld, checksum = %.1f (err = %.6f, tol = %.6f)\n",
               total, cs, err, (double)REF_TOL);
    }

    free(x);
    free(counts);
    MPI_Finalize();
    return 0;
}
