#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scan.h"
#include "reference.h"
#include "problem.h"

int main() {
    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));
    if (!in || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (long i = 0; i < REF_N; i++) {
        in[i] = in_of(i);
    }

    scan(in, out, REF_N);

    double actual_checksum = checksum(out, REF_N);
    double diff = fabs(actual_checksum - REF_CHECKSUM);

    printf("Expected checksum: %f\n", REF_CHECKSUM);
    printf("Actual checksum:   %f\n", actual_checksum);
    printf("Difference:        %f\n", diff);

    if (diff <= REF_TOL) {
        printf("PASS\n");
        free(in);
        free(out);
        return 0;
    } else {
        printf("FAIL\n");
        free(in);
        free(out);
        return 1;
    }
}
