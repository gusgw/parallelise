/*
 * integrate.c — Midpoint-rule estimate of ∫ 4/(1+x²) dx over [a, b].
 *
 * The hot summation loop is parallelised with an OpenMP reduction so that
 * each thread handles only its own share of the n subintervals; the partial
 * sums are combined by the runtime.  The result converges to π when called
 * as integrate(0.0, 1.0, n).
 */

#include "integrate.h"

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
