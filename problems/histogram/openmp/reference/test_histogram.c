#include <stdlib.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the gate
 * checks it at 1 thread and at many -- the counts must be the same either way, so a
 * data race on the bins shows up as a checksum that moves with the thread count. */
int main(void)
{
    long n = REF_N;
    int nbins = (int)REF_NBINS;
    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    histogram(x, n, counts, nbins);

    CHECK_CLOSE(checksum(counts, nbins), REF_CHECKSUM, REF_TOL,
                "histogram weighted checksum matches the oracle");
    long total = 0;
    for (int b = 0; b < nbins; b++) total += counts[b];
    CHECK(total == n, "histogram counts sum to exactly n");

    free(x); free(counts);
    return TEST_SUMMARY();
}
