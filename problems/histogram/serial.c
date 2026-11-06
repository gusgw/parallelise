/* The serial routine of histogram.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 100000, 3000000, 100000000]. */
#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins) {
    // Initialize counts to zero
    for (int i = 0; i < nbins; i++) {
        counts[i] = 0;
    }
    
    // Count values in each bin
    for (long i = 0; i < n; i++) {
        double v = x[i];
        // Compute bin index: floor(v * nbins)
        // Use (int)(v * nbins) which truncates toward zero
        // Since v >= 0, this is equivalent to floor
        int bin = (int)(v * nbins);
        // Clamp to last bin if necessary
        if (bin >= nbins) {
            bin = nbins - 1;
        }
        counts[bin]++;
    }
}
