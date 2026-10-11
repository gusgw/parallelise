#include "kernel_b.h"
#include "reference.h"
#include "problem.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    long n = REF_N;
    double *u = malloc(sizeof(double) * n * n);
    if (!u) { fprintf(stderr, "alloc failed\n"); return 1; }
    kernel_b(u, n, REF_ITERS);
    double checksum_value = checksum(u, n);
    free(u);
    if (fabs(checksum_value - (REF_N * (1 + REF_ITERS))) > REF_TOL) {
        printf("Test failed!\n");
        return 1;
    }
    printf("Test passed!\n");
    return 0;
}
