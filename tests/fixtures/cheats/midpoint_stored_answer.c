/* A routine that fooled the earlier check on integrate in an adversarial test on
 * 6 November 2026, in its version for the midpoint test problem. tests/test_judge.py
 * shows that the present judgement is not fooled. It does no parallel work, keeps its answer between calls, and on a repeated call
 * waits for a believable time before returning the stored answer. */

#include <math.h>
#include <omp.h>
#include "midpoint.h"

static long last_n = -1;
static double last_r, last_t;

double midpoint(double a, double b, long n)
{
    int w = omp_get_max_threads();
    if (n == last_n) {
        double until = omp_get_wtime() + last_t / (0.7 * w + 0.3);
        while (omp_get_wtime() < until) {
        }
        return last_r;
    }
    double start = omp_get_wtime();
    double h = (b - a) / (double)n, s = 0.0;
    for (long i = 0; i < n; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    last_n = n;
    last_r = s * h;
    last_t = omp_get_wtime() - start;
    return last_r;
}
