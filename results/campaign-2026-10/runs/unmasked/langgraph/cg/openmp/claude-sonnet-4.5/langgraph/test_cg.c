#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"

int main(void)
{
    long m = REF_M;
    int iters = REF_ITERS;
    long n = m * m;
    
    /* Allocate and fill right-hand side */
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Allocate solution vector */
    double *x = (double *)malloc(n * sizeof(double));
    
    /* Run the solver */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    double err = fabs(cs - REF_CHECKSUM);
    
    printf("Checksum: %.10f\n", cs);
    printf("Expected: %.10f\n", REF_CHECKSUM);
    printf("Error:    %.10e\n", err);
    
    if (err > REF_TOL) {
        printf("FAIL: error %.10e exceeds tolerance %.10e\n", err, REF_TOL);
        free(b);
        free(x);
        return 1;
    }
    
    printf("PASS\n");
    free(b);
    free(x);
    return 0;
}
