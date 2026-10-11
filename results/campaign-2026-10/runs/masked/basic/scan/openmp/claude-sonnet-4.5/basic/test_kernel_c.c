#include <stdlib.h>
#include <math.h>
#include "kernel_c.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    
    if (!in || !out) {
        free(in);
        free(out);
        return 1;
    }
    
    // Fill input using problem.h helper
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    // Run the kernel
    kernel_c(in, out, n);
    
    // Compute checksum using problem.h helper
    double result = checksum(out, n);
    
    // Check against reference
    CHECK_CLOSE(result, REF_CHECKSUM, REF_TOL, "checksum matches reference");
    
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
