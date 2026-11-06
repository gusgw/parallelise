#include "integrate.h"

double integrate(double a, double b, long n)
{
    const double h = (b - a) / (double)n;
    double sum = 0.0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (long i = 0; i < n; i++) {
        const double x = a + ((double)i + 0.5) * h; /* midpoint of subinterval i */
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * h;
}
