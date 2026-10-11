#include "kernel_d.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double x[REF_M], b[REF_M];
    // Initialize x and b
    for (long i = 0; i < REF_M; i++) {
        x[i] = 1000.0;
        b[i] = 1000.0;
    }
    cgsolve(x, b, REF_M, REF_ITERS);
    double checksum = 0.0;
    for (long i = 0; i < REF_M; i++) {
        checksum += x[i];
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed: checksum = %f, expected %f\n", checksum, REF_CHECKSUM);
        return 1;
    }
    printf("Test passed\n");
    return 0;
}
