#include "scan.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void scan(const double *in, double *out, long n) {
    // parallel inclusive scan
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double sum = 0.0;
        for (long j = 0; j <= i; j++) {
            sum += in[j];
        }
        out[i] = sum;
    }
}
