#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long m = REF_M;
    int iters = REF_ITERS;
    long n = m * m;
    
    /* Allocate and fill b (right-hand side) */
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Allocate solution vector x */
    double *x = (double *)malloc(n * sizeof(double));
    
    /* Call the kernel (collective, all ranks must call) */
    cgsolve(x, b, m, iters);
    
    /* Only rank 0 checks and reports results */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        printf("checksum = %.12g (expected %.12g)\n", cs, REF_CHECKSUM);
    }
    
    free(b);
    free(x);
    
    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }
    
    MPI_Finalize();
    return result;
}
