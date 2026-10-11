#include "kernel_c.h"
#include "problem.h"
#include <omp.h>

void kernel_c(const double *in, double *out, long n) {
    #pragma omp parallel for
    for (long t = 0; t < n; t++) {
        out[t] = in[t] * in[t];
    }
}
