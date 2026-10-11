#include "kernel_c.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long n = REF_N;
    
    // Allocate and fill input
    double *in = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    // Call the kernel
    kernel_c(in, out, n);
    
    // Compute checksum in a distributed way
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    
    double local_cs = 0.0;
    for (long i = lo; i < hi; i++) {
        local_cs += out[i];
    }
    
    double cs = 0.0;
    MPI_Reduce(&local_cs, &cs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Only rank 0 checks the result
    if (rank == 0) {
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
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
