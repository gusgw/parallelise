#include "scan.h"
#include <mpi.h>
#include <stdlib.h>

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Collective routine: every MPI rank calls it together with identical in.
 * The result stays distributed: rank r holds correct values for its own
 * contiguous block out[lo, hi), lo = n*rank/size, hi = n*(rank+1)/size.
 *
 * Block-scan decomposition:
 * 1. Each rank scans its own block of in locally
 * 2. Gather all block totals
 * 3. Each rank computes its offset (sum of all lower ranks' totals)
 * 4. Add the offset to the local block
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
    
    /* Step 2: Gather all block totals (the last element of each block's scan) */
    double my_total = (block_size > 0) ? running_sum : 0.0;
    double *all_totals = (double *)malloc(size * sizeof(double));
    MPI_Allgather(&my_total, 1, MPI_DOUBLE, all_totals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    
    /* Step 3: Compute offset for this rank (sum of all lower ranks' totals) */
    double offset = 0.0;
    for (int r = 0; r < rank; r++) {
        offset += all_totals[r];
    }
    
    /* Step 4: Add offset to this rank's block */
    for (long i = 0; i < block_size; i++) {
        out[lo + i] += offset;
    }
    
    free(all_totals);
}
