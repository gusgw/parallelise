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

    /* Allocate and fill input array with deterministic values */
    double *x = malloc((size_t)n * sizeof(double));
    if (x == NULL) {
        printf("FAIL: could not allocate input array\n");
        return 1;
    }

    for (long i = 0; i < n; i++) {
        x[i] = x_of(i);
    }

    /* Allocate counts array */
    long *counts = calloc((size_t)nbins, sizeof(long));
    if (counts == NULL) {
        free(x);
        printf("FAIL: could not allocate counts array\n");
        return 1;
    }

    /* Run the histogram kernel */
    histogram(x, n, counts, nbins);

    /* Verify total count equals n */
    long total = 0;
    for (int b = 0; b < nbins; b++) {
        total += counts[b];
    }
    CHECK(total == n, "total count should equal n");

    /* Compute and verify checksum */
    double cs = checksum(counts, nbins);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum should match reference");

    free(x);
    free(counts);

    return TEST_SUMMARY();
}
