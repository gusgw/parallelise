#include <stdio.h>
#include <math.h>
#include "nbody.h"
#include "reference.h"
#define N_TEST REF_N
#define CHECKSUM_REF REF_CHECKSUM
#define TOL REF_TOL

int main() {
    double px[N_TEST], py[N_TEST], ax[N_TEST], ay[N_TEST];
    for (int i = 0; i < N_TEST; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }
    nbody_accel(px, py, ax, ay, N_TEST);
    double checksum = 0.0;
    for (int i = 0; i < N_TEST; i++) {
        checksum += fabs(ax[i]) + fabs(ay[i]);
    }
    if (fabs(checksum - CHECKSUM_REF) > TOL) {
        printf("ERROR: checksum mismatch: %f != %f\n", checksum, CHECKSUM_REF);
        return 1;
    }
    printf("OK\n");
    return 0;
}
