#include "scan.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Parallel block-scan decomposition:
 * 1. Each rank scans its own block of in locally
 * 2. Gather all block totals (last element of each block's scan)
 * 3. Each rank computes its starting offset by summing totals from all lower ranks
 * 4. Add the offset to each element in the local block
 */
void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute this rank's block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long block_size = hi - lo;

    /* Step 1: Local inclusive scan of this rank's block */
    double running_sum = 0.0;
    for (long i = 0; i < block_size; i++) {
        running_sum += in[lo + i];
        out[lo + i] = running_sum;
    }

    /* Step 2: Gather all block totals (last element of each block) */
    double *block_totals = NULL;
    if (rank == 0) {
        block_totals = (double *)malloc(size * sizeof(double));
    } else {
        block_totals = (double *)malloc(size * sizeof(double));
    }
    
    /* Each rank's block total is the last element of its scan (running_sum) */
    MPI_Allgather(&running_sum, 1, MPI_DOUBLE, block_totals, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    /* Step 3: Compute offset for this rank (sum of all lower ranks' totals) */
    double offset = 0.0;
    for (int r = 0; r < rank; r++) {
        offset += block_totals[r];
    }

    /* Step 4: Add offset to all elements in this rank's block */
    for (long i = 0; i < block_size; i++) {
        out[lo + i] += offset;
    }

    free(block_totals);
}
