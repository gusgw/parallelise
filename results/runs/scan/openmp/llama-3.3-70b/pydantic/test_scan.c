#include "scan.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    long N = REF_N;
    double *in = (double *)malloc(N * sizeof(double));
    double *out = (double *)malloc(N * sizeof(double));

    // fill input
    for (long i = 0; i < N; i++) {
        in[i] = ((i * 2654435761) % 1000003) / 1000003.0 - 0.5;
    }

    // run scan
    scan(in, out, N);

    // check result
    double checksum = 0.0;
    for (long i = 0; i < N; i++) {
        checksum += out[i];
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }

    printf("SUCCESS\n");
    return 0;
}
