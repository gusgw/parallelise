// kernel_a.c — parallel implementation of the midpoint integration kernel.
// Computes the integral of 4/(1+x^2) over [a,b] using n midpoint samples.
// The implementation follows the reference Python version and uses OpenMP
// to parallelise the summation.

#include "kernel_a.h"
#include <omp.h>
#include <stddef.h>

double kernel_a(double a, double b, long n) {
    if (n <= 0) {
        return 0.0;
    }
    double h = (b - a) / (double)n;
    double sum = 0.0;
    // Parallel reduction over the sum of the integrand values.
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
