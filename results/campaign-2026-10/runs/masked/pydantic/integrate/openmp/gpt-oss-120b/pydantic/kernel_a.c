#include "kernel_a.h"
#include <omp.h>

/*
 * Compute the midpoint Riemann sum of f(x) = 4 / (1 + x^2) over [a, b]
 * with n sub‑intervals. The implementation is parallelised with OpenMP.
 */

double kernel_a(double a, double b, long n)
{
    if (n <= 0) return 0.0;
    double h = (b - a) / (double)n;
    double sum = 0.0;
    /* Parallel reduction over the loop index i */
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
