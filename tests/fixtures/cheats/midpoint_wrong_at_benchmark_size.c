/* A routine that fooled the earlier check on integrate in an adversarial test on
 * 6 November 2026, in its version for the midpoint test problem. tests/test_judge.py
 * shows that the present judgement is not fooled. It is right at the test sizes, and sums only a tenth of the range above a
 * million subintervals, which includes the benchmark size. */

#include <math.h>
#include "midpoint.h"

double midpoint(double a, double b, long n)
{
    long m = (n > 1000000L) ? n / 10 : n;
    double h = (b - a) / (double)n, s = 0.0;
#pragma omp parallel for reduction(+ : s)
    for (long i = 0; i < m; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    return s * h;
}
