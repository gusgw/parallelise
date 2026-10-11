#include "cg.h"
#include "reference.h"
#include "problem.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    
    /* Run the CG solver */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    double err = fabs(cs - REF_CHECKSUM);
    
    printf("Checksum: %.10f (expected %.10f, error %.2e)\n", cs, REF_CHECKSUM, err);
    
    if (err > REF_TOL) {
        fprintf(stderr, "FAIL: error %.2e exceeds tolerance %.2e\n", err, REF_TOL);
        free(b);
        free(x);
        return 1;
    }
    
    printf("PASS\n");
    free(b);
    free(x);
    return 0;
}
