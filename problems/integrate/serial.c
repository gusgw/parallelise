/* The serial routine of integrate.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 100000, 50000000]. */
#include "integrate.h"

double integrate(double a, double b, long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
