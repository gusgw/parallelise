#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cg.h"
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
    long n = m * m;
    
    /* Allocate and fill b (every rank has full copy) */
    double *x = malloc(sizeof(double) * n);
    double *b = malloc(sizeof(double) * n);
    if (!x || !b) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }
    
    /* Call the kernel (collective) */
    cgsolve(x, b, m, iters);
    
    /* Only rank 0 checks and reports */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
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
