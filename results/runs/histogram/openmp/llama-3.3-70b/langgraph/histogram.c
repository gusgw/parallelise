#include "histogram.h"
#include <omp.h>

void histogram(const double *x, long n, long *counts, int nbins) {
    #pragma omp parallel for reduction(+:counts[:nbins])
    for (long i = 0; i < n; i++) {
        long b = (long)(x[i] * nbins);
        if (b >= nbins) b = nbins - 1;
        counts[b]++;
    }
}
