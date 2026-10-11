#include "kernel_c.h"
#include <mpi.h>
#include <stdlib.h>

void kernel_c(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Compute this rank's range
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;
    
    // Allocate local buffer
    double *local_buf = (double *)malloc(local_n * sizeof(double));
    
    // Compute local prefix sum for this rank's chunk
    double run = 0.0;
    for (long i = 0; i < local_n; i++) {
        run += in[lo + i];
        local_buf[i] = run;
    }
    
    // Use MPI_Exscan to get the sum of all elements from previous ranks
    double offset = 0.0;
    MPI_Exscan(&run, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Add the offset to all local elements
    if (rank > 0) {
        for (long i = 0; i < local_n; i++) {
            local_buf[i] += offset;
        }
    }
    
    // Check if all chunks are the same size (for MPI_Gather)
    long chunk_size = n / size;
    int use_gather = (n % size == 0);
    
    if (use_gather) {
        // Use MPI_Gather for equal-sized chunks
        if (rank == 0) {
            MPI_Gather(local_buf, (int)chunk_size, MPI_DOUBLE,
                      out, (int)chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(local_buf, (int)chunk_size, MPI_DOUBLE,
                      NULL, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
        }
    } else {
        // Use MPI_Gatherv for unequal chunks
        int *recvcounts = NULL;
        int *displs = NULL;
        
        if (rank == 0) {
            recvcounts = (int *)malloc(size * sizeof(int));
            displs = (int *)malloc(size * sizeof(int));
            
            for (int r = 0; r < size; r++) {
                long r_lo = n * r / size;
                long r_hi = n * (r + 1) / size;
                recvcounts[r] = (int)(r_hi - r_lo);
                displs[r] = (int)r_lo;
            }
            
            MPI_Gatherv(local_buf, (int)local_n, MPI_DOUBLE,
                       out, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            free(recvcounts);
            free(displs);
        } else {
            MPI_Gatherv(local_buf, (int)local_n, MPI_DOUBLE,
                       NULL, NULL, NULL, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
        }
    }
    
    free(local_buf);
}
