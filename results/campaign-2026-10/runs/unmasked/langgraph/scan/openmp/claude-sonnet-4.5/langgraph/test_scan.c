#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long n = REF_N;
    
    /* Allocate input and output arrays */
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    
    /* Fill input using the deterministic pseudo-random fill */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Run the scan kernel */
    scan(in, out, n);
    
    /* Compute the checksum */
    double cs = checksum(out, n);
    
    /* Check against the reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "scan checksum");
    
    /* Clean up */
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
