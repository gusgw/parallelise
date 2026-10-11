#include "histogram.h"
#include <omp.h>

void histogram(const double *x, long n, long *counts, int nbins) {
    #pragma omp parallel
    {
        long local_counts[nbins];
        #pragma omp for
        for (int i = 0; i < nbins; i++) {
            local_counts[i] = 0;
        }
        #pragma omp for
        for (long i = 0; i < n; i++) {
            int b = (int)(x[i] * nbins);
            if (b >= nbins) {
                b = nbins - 1;
            }
            local_counts[b]++;
        }
        #pragma omp critical
        {
            for (int i = 0; i < nbins; i++) {
                counts[i] += local_counts[i];
            }
        }
    }
}
