#include "histogram.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>

int main() {
    double x[REF_N];
    long counts[REF_NBINS];
    for (long i = 0; i < REF_N; i++) {
        x[i] = ((i * 2654435761 + 12345) % 4294967296) / 4294967296.0;
    }
    histogram(x, REF_N, counts, REF_NBINS);
    double checksum = 0.0;
    for (int b = 0; b < REF_NBINS; b++) {
        checksum += counts[b] * (b + 1);
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
