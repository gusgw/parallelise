#include "histogram.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    long n = REF_N;
    int nbins = REF_NBINS;
    double *x = (double *)malloc(n * sizeof(double));
    long *counts = (long *)malloc(nbins * sizeof(long));
    for (long i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX;
    }
    histogram(x, n, counts, nbins);
    double checksum = 0.0;
    for (int b = 0; b < nbins; b++) {
        checksum += counts[b] * (b + 1);
    }
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    return 0;
}
