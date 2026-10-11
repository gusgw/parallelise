#include "scan.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() {
    int n = REF_N;
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    double checksum = 0.0;
    srand(time(NULL));

    // Fill input array
    for (int i = 0; i < n; i++) {
        in[i] = ((i * 2654435761) % 1000003) / 1000003.0 - 0.5;
    }

    // Run scan
    scan(in, out, n);

    // Compute checksum
    for (int i = 0; i < n; i++) {
        checksum += out[i];
    }

    // Check against reference
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Checksum mismatch: %f != %f\n", checksum, REF_CHECKSUM);
        return 1;
    }

    printf("Checksum match: %f == %f\n", checksum, REF_CHECKSUM);
    return 0;
}
