#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_d.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    double *x = malloc(sizeof(double) * n);
    double *b = malloc(sizeof(double) * n);
    
    if (!x || !b) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Fill b using problem.h helper */
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }
    
    /* Initialize x to zero */
    for (long t = 0; t < n; t++) {
        x[t] = 0.0;
    }
    
    /* Call the kernel - all ranks participate */
    cgsolve(x, b, m, iters);
    
    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        printf("checksum = %.12f (expected %.12f)\n", cs, REF_CHECKSUM);
    }
    
    free(x);
    free(b);
    
    int failed = 0;
    if (rank == 0) {
        failed = TEST_SUMMARY();
    }
    
    MPI_Finalize();
    return failed;
}
