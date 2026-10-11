#include <math.h>

#include "midpoint.h"

/* Midpoint rule for sin(x); the reduction gives each thread its own partial sum. */
double midpoint(double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double integral_sum = 0.0;
#pragma omp parallel for reduction(+ : integral_sum)
    for (long i = 0; i < n; i++) {
        integral_sum += sin(a + ((double)i + 0.5) * h);
    }
    return integral_sum * h;
}
