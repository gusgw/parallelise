#include <omp.h>
#include <math.h>
#include "integrate.h"

// Function to perform numerical integration using the midpoint rule
void integrate(double a, double b, int n, double* result) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000)
    for (int i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    *result = sum * h;
}
