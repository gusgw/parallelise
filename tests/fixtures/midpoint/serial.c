/* The serial routine for the midpoint fixture, written by hand for the tests. */
#include <math.h>

#include "midpoint.h"

double midpoint(double a, double b, long n)
{
    const double h = (b - a) / (double)n;
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        sum += sin(a + ((double)i + 0.5) * h);
    }
    return sum * h;
}
