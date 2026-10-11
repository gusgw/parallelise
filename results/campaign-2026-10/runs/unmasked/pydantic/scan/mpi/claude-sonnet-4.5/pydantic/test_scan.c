#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long n = REF_N;
    
    /* Allocate and fill input array (same on all ranks) */
    double *in = (double *)malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Allocate output array */
    double *out = (double *)malloc(n * sizeof(double));
    
    /* Call the kernel (collective) */
    scan(in, out, n);
    
    /* Compute this rank's block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    
    /* Compute partial checksum for this rank's block */
    double partial_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        partial_sum += out[i];
    }
    
    /* Reduce all partial checksums to get total checksum */
    double total_checksum = 0.0;
    MPI_Reduce(&partial_sum, &total_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    /* Only rank 0 checks and reports */
    if (rank == 0) {
        CHECK_CLOSE(total_checksum, REF_CHECKSUM, REF_TOL, "checksum matches reference");
        printf("checksum = %.12g (expected %.12g, tol %g)\n", 
               total_checksum, REF_CHECKSUM, REF_TOL);
    }
    
    free(in);
    free(out);
    
    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }
    
    MPI_Finalize();
    return result;
}
