#include "kernel_c.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    double *in = (double *) malloc(REF_N * sizeof(double));
    double *out = (double *) malloc(REF_N * sizeof(double));
    long i;
    for (i = 0; i < REF_N; i++) {
        in[i] = (double) i;
    }
    kernel_c(in, out, REF_N);
    double checksum = 0;
    for (i = 0; i < REF_N; i++) {
        checksum += out[i];
    }
    printf("Checksum: %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    } else {
        printf("Test passed!\n");
        return 0;
    }
}
