#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reference.h"
#include "problem.h"

int main() {
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(sizeof(double) * n * n);
    double *unew = malloc(sizeof(double) * n * n);
    
    // Initialize
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    printf("Initial checksum: %f\n", checksum(u, n));
    
    // Jacobi iterations
    for (int iter = 0; iter < iters; iter++) {
        memcpy(unew, u, sizeof(double) * n * n);
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                unew[i * n + j] = 0.25 * (u[(i - 1) * n + j] +
                                          u[(i + 1) * n + j] +
                                          u[i * n + (j - 1)] +
                                          u[i * n + (j + 1)]);
            }
        }
        double *tmp = u;
        u = unew;
        unew = tmp;
    }
    
    double cs = checksum(u, n);
    printf("Final checksum: %.12f\n", cs);
    printf("Expected: %.12f\n", REF_CHECKSUM);
    printf("Diff: %g\n", cs - REF_CHECKSUM);
    
    free(u);
    free(unew);
    return 0;
}
