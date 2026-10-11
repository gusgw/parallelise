#include "scan.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double *in = (double *) malloc(REF_N * sizeof(double));
    double *out = (double *) malloc(REF_N * sizeof(double));
    // Fill in with some values for testing
    for (long i = 0; i < REF_N; i++) {
        in[i] = i;
    }
    scan(in, out, REF_N);
    double checksum = 0;
    for (long i = 0; i < REF_N; i++) {
        checksum += out[i];
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed: checksum %f, expected %f\n", checksum, REF_CHECKSUM);
        return 1;
    }
    printf("Test passed\n");
    free(in);
    free(out);
    return 0;
}
