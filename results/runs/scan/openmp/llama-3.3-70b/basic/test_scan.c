#include "scan.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    double *in = (double *) malloc(REF_N * sizeof(double));
    double *out = (double *) malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        in[i] = (i * 2654435761) % 1000003 / 1000003.0 - 0.5;
    }
    scan(in, out, REF_N);
    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        checksum += out[i];
    }
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    printf("OK\n");
    free(in);
    free(out);
    return 0;
}
