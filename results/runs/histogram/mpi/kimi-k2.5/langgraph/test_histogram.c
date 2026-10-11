#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "histogram.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Allocate and fill the input array deterministically */
    long n = REF_N;
    int nbins = (int)REF_NBINS;
    double *x = (double *)malloc(n * sizeof(double));
    if (x == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Failed to allocate input array\n");
        }
        MPI_Finalize();
        return 1;
    }

    for (long i = 0; i < n; i++) {
        x[i] = x_of(i);
    }

    /* Allocate output histogram */
    long *counts = (long *)malloc(nbins * sizeof(long));
    if (counts == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Failed to allocate counts array\n");
        }
        free(x);
        MPI_Finalize();
        return 1;
    }

    /* Call the kernel - collective operation, all ranks participate */
    histogram(x, n, counts, nbins);

    /* Compute checksum and verify on rank 0 */
    if (rank == 0) {
        double cs = checksum(counts, nbins);

        /* Verify checksum matches reference */
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

        /* Verify total count equals n */
        long total = 0;
        for (int b = 0; b < nbins; b++) {
            total += counts[b];
        }
        CHECK(total == n, "total count equals n");

        int result = TEST_SUMMARY();
        free(counts);
        free(x);
        MPI_Finalize();
        return result;
    }

    /* Non-rank-0 ranks just clean up */
    free(counts);
    free(x);
    MPI_Finalize();
    return 0;
}
