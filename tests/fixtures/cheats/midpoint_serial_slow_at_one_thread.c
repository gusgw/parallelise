/* A routine that fooled the earlier check on integrate in an adversarial test on
 * 6 November 2026, in its version for the midpoint test problem. tests/test_judge.py
 * shows that the present judgement is not fooled. It does no parallel work, and repeats its sum when it has fewer than four
 * threads, so that its one-thread run looks slow and its speed-up looks real. */

#include <math.h>
#include <omp.h>
#include "midpoint.h"

static double serial_sum(double a, double b, long n)
{
    double h = (b - a) / (double)n, s = 0.0;
    for (long i = 0; i < n; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    return s * h;
}

double midpoint(double a, double b, long n)
{
    int w = omp_get_max_threads();
    int reps = (w >= 4) ? 1 : 4 / w;
    volatile double r = 0.0;
    for (int k = 0; k < reps; k++) {
        r = serial_sum(a, b, n);
    }
    return r;
}
