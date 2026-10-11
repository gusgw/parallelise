#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "histogram.h"
#include "reference.h"
#include "problem.h"

int main() {
    double *x = (double *)malloc(REF_N * sizeof(double));
    long *counts = (long *)malloc(REF_NBINS * sizeof(long));

    for (long i = 0; i < REF_N; i++) {
        x[i] = x_of(i);
    }

    histogram(x, REF_N, counts, (int)REF_NBINS);

    double cs = checksum(counts, (int)REF_NBINS);
    printf("Checksum: %f (Expected: %f)\n", cs, REF_CHECKSUM);

    if (fabs(cs - REF_CHECKSUM) <= REF_TOL) {
        printf("PASS\n");
        free(x);
        free(counts);
        return 0;
    } else {
        printf("FAIL\n");
        free(x);
        free(counts);
        return 1;
    }
}
