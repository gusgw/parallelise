#include "kernel_c.h"
#include "reference.h"
#include "problem.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long n = REF_N;
    
    // Allocate and fill input
    double *in = (double *)malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    // Allocate output
    double *out = (double *)malloc(n * sizeof(double));
    
    // Call the kernel
    kernel_c(in, out, n);
    
    // Only rank 0 checks and prints
    if (rank == 0) {
        double cs = checksum(out, n);
        double err = fabs(cs - REF_CHECKSUM);
        
        if (err < REF_TOL) {
            printf("PASS: checksum=%.10f, expected=%.10f, error=%.10e\n",
                   cs, REF_CHECKSUM, err);
        } else {
            printf("FAIL: checksum=%.10f, expected=%.10f, error=%.10e (tol=%.10e)\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
            free(in);
            free(out);
            MPI_Finalize();
            return 1;
        }
    }
    
    free(in);
    free(out);
    
    MPI_Finalize();
    return 0;
}
