#include "integrate.h"

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals.  On [0, 1] this converges to pi.
 *
 * Algorithm (direct port from integrate.ipynb):
 *   h = (b - a) / n
 *   s = sum_{i=0}^{n-1}  4 / (1 + x_i^2),   x_i = a + (i + 0.5) * h
 *   return s * h
 *
 * The hot loop is parallelised with an OpenMP reduction so each thread
 * accumulates its own partial sum; the runtime combines them at the end.
 */
double integrate(double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double s = 0.0;

#pragma omp parallel for reduction(+:s) schedule(static)
    for (long i = 0; i < n; i++) {
        double x = a + ((double)i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }

    return s * h;
}
