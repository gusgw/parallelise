#include "scan.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute local block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Step 1: Local inclusive scan of the block, keeping track of total */
    double local_sum = 0.0;
    for (long i = 0; i < local_n; i++) {
        local_sum += in[lo + i];
        out[lo + i] = local_sum;
    }

    /* Step 2: Use MPI_Scan to get the sum of all previous ranks' blocks */
    /* MPI_Scan does an inclusive prefix sum, so we need the sum of all PREVIOUS ranks */
    /* We use MPI_Exscan which gives exclusive prefix sum (sum of all previous) */
    double my_offset = 0.0;
    MPI_Exscan(&local_sum, &my_offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* Note: rank 0 gets MPI_PROC_NULL result which is 0.0, which is correct */

    /* Step 3: Add offset to local block */
    for (long i = 0; i < local_n; i++) {
        out[lo + i] += my_offset;
    }
}
