/*
 * integrate.c – OpenMP parallel implementation of the midpoint rule for
 * integrating the function f(x) = 4 / (1 + x^2) over an interval [a, b].
 *
 * The serial algorithm (see integrate.ipynb) computes the step size h = (b-a)/n
 * and then evaluates the integrand at the midpoint of each sub‑interval:
 *   x_i = a + (i + 0.5) * h   for i = 0 … n-1
 * The sum of the function values is multiplied by h to obtain the integral
 * estimate.
 *
 * This file provides a thread‑parallel version using OpenMP.  The loop over the
 * sub‑intervals is split across threads with a reduction on the accumulator.
 * The implementation follows the signature declared in integrate.h and
 * compiles with -fopenmp and -Wall -Wextra without warnings.
 */

#include "integrate.h"
#include <math.h>
#include <omp.h>

double integrate(double a, double b, long n)
{
    /* Guard against degenerate input – the reference test never triggers this,
     * but returning 0.0 makes the function well‑behaved for n <= 0. */
    if (n <= 0) {
        return 0.0;
    }

    const double h = (b - a) / (double)n;
    double sum = 0.0;

    /* Parallelise the summation.  The reduction clause ensures each thread
     * contributes its partial sum to the final result without race conditions.
     */
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        const double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * h;
}
