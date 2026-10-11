#include "kernel_c.h"
#include <mpi.h>

void kernel_c(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Divide work: each rank gets a contiguous block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;
    
    /* Compute local prefix sum directly into output buffer */
    double running_sum = 0.0;
    for (long i = 0; i < local_n; i++) {
        running_sum += in[lo + i];
        out[lo + i] = running_sum;
    }
    
    /* Get the cumulative sum up to and including this rank using MPI_Scan */
    double cumulative_sum = 0.0;
    if (size > 1) {
        MPI_Scan(&running_sum, &cumulative_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else {
        cumulative_sum = running_sum;
    }
    
    /* The offset from previous ranks is cumulative_sum - running_sum */
    double previous_sum = cumulative_sum - running_sum;
    
    /* Add the offset from previous ranks to all local elements */
    for (long i = 0; i < local_n; i++) {
        out[lo + i] += previous_sum;
    }
}
