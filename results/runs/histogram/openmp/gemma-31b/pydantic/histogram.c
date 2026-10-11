#include "histogram.h"
#include <string.h>
#include <omp.h>
#include <stdlib.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    memset(counts, 0, nbins * sizeof(long));

    #pragma omp parallel
    {
        long *local_counts = (long *)calloc(nbins, sizeof(long));
        if (local_counts) {
            #pragma omp for nowait
            for (long i = 0; i < n; i++) {
                int b = (int)(x[i] * nbins);
                if (b >= nbins) {
                    b = nbins - 1;
                }
                if (b < 0) {
                    b = 0;
                }
                local_counts[b]++;
            }

            #pragma omp critical
            {
                for (int b = 0; b < nbins; b++) {
                    counts[b] += local_counts[b];
                }
            }
            free(local_counts);
        }
    }
}
