#include "kernel_c.h"
#include "reference.h"
#include "problem.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));
    for (long t = 0; t < REF_N; t++) in[t] = in_of(t);
    kernel_c(in, out, REF_N);
    double checksum_result = checksum(out, REF_N);
    printf("Checksum: %f\n", checksum_result);
    if (fabs(checksum_result - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    }
    printf("Test passed!\n");
    free(in);
    free(out);
    return 0;
}
