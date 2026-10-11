#include "scan.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void scan(const double *in, double *out, long n) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        out[i] = in[i];
    }
    for (int len = 1; len < n; len *= 2) {
        #pragma omp parallel for
        for (i = len; i < n; i++) {
            out[i] += out[i - len];
        }
    }
}
