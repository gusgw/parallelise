#include "kernel_b.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "problem.h"

int main() {
    double *u = (double *)malloc(REF_N * REF_N * sizeof(double));
    kernel_b(u, REF_N, REF_ITERS);
    double checksum_result = checksum(u, REF_N);
    printf("Checksum: %f\n", checksum_result);
    if (fabs(checksum_result - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    }
    printf("Test passed!\n");
    free(u);
    return 0;
}
