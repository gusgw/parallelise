#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int nbins = (int)REF_NBINS;

    /* Allocate and fill input array with deterministic values */
    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);
    if (!x || !counts) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        x[i] = x_of(i);
    }

    /* Call the kernel (collective - all ranks call it) */
    histogram(x, n, counts, nbins);

    /* Verify results on rank 0 */
    if (rank == 0) {
        /* Check total count */
        long total = 0;
        for (int b = 0; b < nbins; b++) {
            total += counts[b];
        }
        CHECK(total == n, "total count equals n");

        /* Check checksum against reference */
        double cs = checksum(counts, nbins);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");

        int result = TEST_SUMMARY();
        free(x);
        free(counts);
        MPI_Finalize();
        return result;
    }

    free(x);
    free(counts);
    MPI_Finalize();
    return 0;
}
