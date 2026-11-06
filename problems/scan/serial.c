/* The serial routine of scan.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 1000000, 50000000]. */
#include "scan.h"

void scan(const double *x, double *out, long n) {
    if (n <= 0) {
        return;
    }
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        sum += x[i];
        out[i] = sum;
    }
}
