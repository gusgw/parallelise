#include "kernel_b.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long n = REF_N;
    int iters = REF_ITERS;
    
    // Allocate and initialize u
    double *u = (double *)malloc(n * n * sizeof(double));
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    // Run the kernel
    kernel_b(u, n, iters);
    
    // Compute checksum
    double cs = checksum(u, n);
    
    // Check against reference
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(u);
    return TEST_SUMMARY();
}
