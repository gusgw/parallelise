#include "kernel_c.h"
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
    
    /* Allocate and fill input array */
    double *in = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Call the kernel */
    kernel_c(in, out, n);
    
    /* Gather all results to rank 0 */
    if (size > 1) {
        int *recvcounts = NULL;
        int *displs = NULL;
        double *gathered_out = NULL;
        
        if (rank == 0) {
            recvcounts = malloc(size * sizeof(int));
            displs = malloc(size * sizeof(int));
            gathered_out = malloc(n * sizeof(double));
            
            for (int r = 0; r < size; r++) {
                long r_lo = n * r / size;
                long r_hi = n * (r + 1) / size;
                recvcounts[r] = (int)(r_hi - r_lo);
                displs[r] = (int)r_lo;
            }
        }
        
        long lo = n * rank / size;
        long hi = n * (rank + 1) / size;
        long local_n = hi - lo;
        
        MPI_Gatherv(&out[lo], (int)local_n, MPI_DOUBLE,
                    rank == 0 ? gathered_out : NULL,
                    recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double cs = checksum(gathered_out, n);
            CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
            free(gathered_out);
            free(recvcounts);
            free(displs);
        }
    } else {
        /* Single rank: check directly */
        double cs = checksum(out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    }
    
    /* Only rank 0 reports results */
    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }
    
    free(in);
    free(out);
    
    MPI_Finalize();
    return result;
}
