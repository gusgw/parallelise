#include "integrate.h"
#include <omp.h>

double integrate(double a, double b, long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * h;
}
