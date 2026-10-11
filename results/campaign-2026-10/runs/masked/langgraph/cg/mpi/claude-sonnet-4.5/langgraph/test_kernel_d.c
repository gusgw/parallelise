#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long m = REF_M;
    int iters = REF_ITERS;
    
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    
    if (!x || !b) {
        if (rank == 0) {
            fprintf(stderr, "alloc failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize b on all ranks
    for (long t = 0; t < m * m; t++) {
        b[t] = b_of(t);
    }
    
    // Call the kernel
    cgsolve(x, b, m, iters);
    
    // Only rank 0 checks and reports
    int result = 0;
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        result = TEST_SUMMARY();
    }
    
    free(x);
    free(b);
    
    MPI_Finalize();
    return result;
}
