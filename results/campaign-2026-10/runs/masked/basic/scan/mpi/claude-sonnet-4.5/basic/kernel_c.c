#include "kernel_c.h"
#include <mpi.h>

void kernel_c(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Compute the range for this rank
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;
    
    // Compute local prefix sum
    double run = 0.0;
    for (long i = 0; i < local_n; i++) {
        run += in[lo + i];
        out[lo + i] = run;
    }
    
    // Get the cumulative sum up to and including this rank
    double cumsum = 0.0;
    MPI_Scan(&run, &cumsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // The offset is the cumulative sum minus the local sum
    double offset = cumsum - run;
    
    // Add offset to all local elements
    for (long i = 0; i < local_n; i++) {
        out[lo + i] += offset;
    }
}
