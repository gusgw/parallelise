#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    
    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        free(in);
        free(out);
        return 1;
    }
    
    /* Fill input array using deterministic pseudo-random fill */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Run the scan kernel */
    scan(in, out, n);
    
    /* Compute checksum */
    double cs = checksum(out, n);
    
    /* Verify against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
