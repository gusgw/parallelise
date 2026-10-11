#include <stdio.h>
#include <stdlib.h>

#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    int nbins = (int)REF_NBINS;

    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);

    if (!x || !counts) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Fill with deterministic pseudo-random values */
    for (long i = 0; i < n; i++) {
        x[i] = x_of(i);
    }

    /* Compute histogram */
    histogram(x, n, counts, nbins);

    /* Verify total count */
    long total = 0;
    for (int b = 0; b < nbins; b++) {
        total += counts[b];
    }
    CHECK(total == n, "total count should equal n");

    /* Verify checksum against reference */
    double cs = checksum(counts, nbins);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum should match reference");

    free(x);
    free(counts);

    return TEST_SUMMARY();
}
