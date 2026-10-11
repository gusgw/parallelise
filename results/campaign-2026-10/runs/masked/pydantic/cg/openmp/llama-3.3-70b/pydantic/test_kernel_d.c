#include "kernel_d.h"
#include "reference.h"
#include "problem.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    double *x = (double *)malloc(REF_M * REF_M * sizeof(double));
    double *b = (double *)malloc(REF_M * REF_M * sizeof(double));

    for (long t = 0; t < REF_M * REF_M; t++) b[t] = b_of(t);

    cgsolve(x, b, REF_M, REF_ITERS);

    double checksum_result = checksum(x, REF_M);

    printf("Checksum: %f\n", checksum_result);

    if (fabs(checksum_result - REF_CHECKSUM) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    }

    printf("Test passed!\n");
    return 0;
}
