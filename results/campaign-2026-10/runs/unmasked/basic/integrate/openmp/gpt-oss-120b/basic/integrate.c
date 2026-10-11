#include "integrate.h"
#include <math.h>

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals. Parallelised with OpenMP reduction.
 */

double integrate(double a, double b, long n) {
    if (n <= 0) {
        return 0.0;
    }
    double h = (b - a) / (double)n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        double x = a + ( (double)i + 0.5 ) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
