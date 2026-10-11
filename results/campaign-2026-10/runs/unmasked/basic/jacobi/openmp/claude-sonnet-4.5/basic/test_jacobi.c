#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>
#include <math.h>

int main(void)
{
    /* Allocate and initialize the grid */
    double *u = (double *)malloc(REF_N * REF_N * sizeof(double));
    
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = u_init(i, j, REF_N);
        }
    }
    
    /* Run the Jacobi iteration */
    jacobi(u, REF_N, REF_ITERS);
    
    /* Compute the checksum */
    double cs = checksum(u, REF_N);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(u);
    
    return TEST_SUMMARY();
}
