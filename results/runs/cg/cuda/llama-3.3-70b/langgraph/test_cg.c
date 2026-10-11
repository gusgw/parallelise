#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>

int main() {
    float b[M_TEST * M_TEST];
    for (int i = 0; i < M_TEST * M_TEST; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }
    float x[M_TEST * M_TEST];
    float checksum = cg(x, b, M_TEST, ITERS);
    printf("checksum = %f\n", checksum);
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    return 0;
}