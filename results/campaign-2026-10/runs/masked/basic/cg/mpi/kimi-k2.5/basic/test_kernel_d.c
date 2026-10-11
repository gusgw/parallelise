#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long m = REF_M;
    int iters = (int)REF_ITERS;
    
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    
    if (!x || !b) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Initialize b */
    for (long t = 0; t < m * m; t++) {
        b[t] = b_of(t);
    }
    
    /* Call the kernel - all ranks participate */
    cgsolve(x, b, m, iters);
    
    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        printf("checksum = %.12f\n", cs);
        int result = TEST_SUMMARY();
        free(x);
        free(b);
        MPI_Finalize();
        return result;
    }
    
    free(x);
    free(b);
    MPI_Finalize();
    return 0;
}
