#include <stdio.h>
#include <math.h>
#include "nbody.h"
#include "reference.h"

int main() {
    double px[REF_N], py[REF_N], ax[REF_N], ay[REF_N];
    for (int i = 0; i < REF_N; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }
    nbody_accel(px, py, ax, ay, REF_N);
    double checksum = 0.0;
    for (int i = 0; i < REF_N; i++) {
        checksum += fabs(ax[i]) + fabs(ay[i]);
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Checksum mismatch: %f != %f\n", checksum, REF_CHECKSUM);
        return 1;
    }
    printf("Checksum match: %f\n", checksum);
    return 0;
}
