#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute local range */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Compute local prefix sum */
    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        local_sum += in[i];
    }

    /* Compute exclusive prefix sum of local sums across ranks */
    double offset = 0.0;
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0;
    }

    /* Compute local prefix sum with offset, store in out */
    double run = offset;
    for (long i = lo; i < hi; i++) {
        run += in[i];
        out[i] = run;
    }

    /* Gather results to rank 0 using a tree-based reduction */
    /* First, gather counts */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
    }
    
    int local_n_int = (int)local_n;
    MPI_Gather(&local_n_int, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(out + lo, local_n_int, MPI_DOUBLE,
                out, recvcounts, displs, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}
