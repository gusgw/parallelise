#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "jacobi.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    /* Allocate and initialize grid */
    double *u = (double *)malloc(n * n * sizeof(double));
    
    /* Initialize: top row hot (1.0), everything else cold (0.0) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    /* Run Jacobi iteration (collective call) */
    jacobi(u, n, iters);
    
    /* Only rank 0 checks and reports results */
    if (rank == 0) {
        double cs = checksum(u, n);
        double err = fabs(cs - REF_CHECKSUM);
        
        printf("checksum = %.16f\n", cs);
        printf("expected = %.16f\n", REF_CHECKSUM);
        printf("error    = %.16f\n", err);
        printf("tol      = %.16f\n", REF_TOL);
        
        if (err <= REF_TOL) {
            printf("PASS: error within tolerance\n");
        } else {
            printf("FAIL: error exceeds tolerance\n");
        }
        
        /* Assert for test framework */
        if (err > REF_TOL) {
            free(u);
            MPI_Finalize();
            return 1;
        }
    }
    
    free(u);
    MPI_Finalize();
    return 0;
}
