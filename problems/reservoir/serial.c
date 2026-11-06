/* The serial routine of reservoir.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 100000, 60000000]. */
#include "reservoir.h"

static const double CAPACITY = 20000.0;

void reservoir(const double *a, double *out, long n) {
    double level = 0.0;
    for (long i = 0; i < n; i++) {
        double sum = level + a[i];
        if (sum < 0.0) {
            sum = 0.0;
        }
        if (sum > CAPACITY) {
            sum = CAPACITY;
        }
        level = sum;
        out[i] = level;
    }
}
