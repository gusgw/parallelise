#include "kernel_c.h"
#include <omp.h>

void kernel_c(const double *in, double *out, long n) {
    long i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        out[i] = in[i] * 2.0;
    }
}
