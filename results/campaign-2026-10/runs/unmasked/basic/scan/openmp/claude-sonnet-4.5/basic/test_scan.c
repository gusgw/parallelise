#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long n = REF_N;
    double *in, *out;
    
    /* Allocate input and output arrays */
    in = (double *)malloc(n * sizeof(double));
    out = (double *)malloc(n * sizeof(double));
    
    /* Fill input using the deterministic pseudo-random fill */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Run the scan */
    scan(in, out, n);
    
    /* Compute checksum and compare to reference */
    double cs = checksum(out, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "scan checksum");
    
    /* Clean up */
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
