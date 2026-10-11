#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel_d.h"
#include "problem.h"
#include "reference.h"

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
        if (rank == 0) fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (long t = 0; t < m * m; t++) {
        b[t] = b_of(t);
    }
    
    cgsolve(x, b, m, iters);
    
    if (rank == 0) {
        double result = checksum(x, m);
        double error = fabs(result - REF_CHECKSUM);
        
        if (error < REF_TOL) {
            printf("PASS: checksum=%.10f, expected=%.10f, error=%.10f\n", 
                   result, REF_CHECKSUM, error);
        } else {
            printf("FAIL: checksum=%.10f, expected=%.10f, error=%.10f\n", 
                   result, REF_CHECKSUM, error);
        }
    }
    
    free(x);
    free(b);
    
    MPI_Finalize();
    return 0;
}
