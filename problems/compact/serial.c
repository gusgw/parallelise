/* The serial routine of compact.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 1000000, 60000000]. */
#include "compact.h"

void compact(const double *x, double threshold, long *indices, long *indices_count, long n) {
    long count = 0;
    for (long i = 0; i < n; i++) {
        if (x[i] > threshold) {
            indices[count] = i;
            count++;
        }
    }
    *indices_count = count;
}
