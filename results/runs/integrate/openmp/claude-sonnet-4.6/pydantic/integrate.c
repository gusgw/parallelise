/* integrate.c — midpoint-rule estimate of ∫ 4/(1+x²) dx over [a, b].
 *
 * The subinterval sum is parallelised with an OpenMP reduction so the result
 * is identical regardless of the number of threads. */

#include "integrate.h"

double integrate(double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double s = 0.0;

#pragma omp parallel for reduction(+:s) schedule(static)
    for (long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }

    return s * h;
}
