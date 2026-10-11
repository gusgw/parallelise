#include "kernel_a.h"
#include <omp.h>

/*
 * Compute the integral of 4/(1+x^2) over [a, b] using the midpoint rule with n
 * sub‑intervals. The implementation is parallelised with OpenMP.
 */

double kernel_a(double a, double b, long n) {
    double h = (b - a) / (double)n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
