// Parallel implementation of prefix-sum kernel.
// Computes out[i] = sum_{j=0}^{i} in[j] for i = 0..n-1.
// Each MPI rank computes only its own slice and applies the exclusive
// scan offset. This version uses pointer arithmetic to reduce indexing overhead.

#include <mpi.h>
#include <stdlib.h>
#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the slice owned by this rank.
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    const double *in_ptr = in + lo;
    double *out_ptr = out + lo;

    // Compute local prefix sum.
    double run = 0.0;
    for (long i = 0; i < local_n; ++i) {
        run += in_ptr[i];
        out_ptr[i] = run;
    }
    double local_sum = run;

    // Exclusive scan to get offset for this rank.
    double offset = 0.0;
    MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0;
    }

    // Apply offset.
    for (long i = 0; i < local_n; ++i) {
        out_ptr[i] += offset;
    }
    // No further communication; each rank holds its slice of the result.
}
