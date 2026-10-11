#include "kernel_b.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>

int main() {
    double u[REF_N];
    for (int i = 0; i < REF_N; i++) {
        u[i] = i; // initialize u
    }
    kernel_b(u, REF_N, REF_ITERS);
    double checksum = 0;
    for (int i = 0; i < REF_N; i++) {
        checksum += u[i];
    }
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    } else {
        printf("Test passed!\n");
        return 0;
    }
}
