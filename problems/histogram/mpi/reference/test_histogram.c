#include <mpi.h>
#include <stdlib.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). histogram() is collective, so every
 * rank calls it (the same number of times, with the same arguments); only rank 0
 * checks the counts against the oracle and reports. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int nbins = (int)REF_NBINS;
    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    histogram(x, n, counts, nbins);

    int rc = 0;
    if (rank == 0) {
        CHECK_CLOSE(checksum(counts, nbins), REF_CHECKSUM, REF_TOL,
                    "histogram weighted checksum matches the oracle");
        long total = 0;
        for (int b = 0; b < nbins; b++) total += counts[b];
        CHECK(total == n, "histogram counts sum to exactly n");
        rc = TEST_SUMMARY();
    }
    free(x); free(counts);
    MPI_Finalize();
    return rc;
}
