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

    /* Allocate and fill the input array (every rank gets the full array) */
    long n = REF_N;
    int nbins = (int)REF_NBINS;
    double *x = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        x[i] = x_of(i);
    }

    /* Allocate output counts */
    long *counts = malloc(nbins * sizeof(long));

    /* Call the kernel (collective - all ranks call it) */
    histogram(x, n, counts, nbins);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        /* Check total count */
        long total = 0;
        for (int b = 0; b < nbins; b++) {
            total += counts[b];
        }
        CHECK(total == n, "total count should equal n");

        /* Check checksum against reference */
        double cs = checksum(counts, nbins);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum should match reference");

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
