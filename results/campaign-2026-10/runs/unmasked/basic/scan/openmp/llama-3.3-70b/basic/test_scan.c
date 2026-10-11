#include "scan.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *in = (double *) malloc(REF_N * sizeof(double));
    double *out = (double *) malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        in[i] = i;
    }
    scan(in, out, REF_N);
    double checksum = 0;
    for (long i = 0; i < REF_N; i++) {
        checksum += out[i];
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Error: checksum mismatch\n");
        return 1;
    }
    printf("Success\n");
    free(in);
    free(out);
    return 0;
}
