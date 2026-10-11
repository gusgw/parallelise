#include "kernel_a.h"
#include <omp.h>

double kernel_a(double a, double b, long n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        double x = a + (b - a) * i / (n - 1);
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * (b - a) / n;
}
