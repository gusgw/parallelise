/* scan.c — parallel inclusive prefix sum (MPI collective).
 *
 * Strategy (classic block-scan decomposition):
 *   1. Each rank computes a local inclusive scan over its contiguous block
 *      of the input [lo, hi).
 *   2. MPI_Exscan(MPI_SUM) of each rank's block total gives the exclusive
 *      prefix offset that must be added to every element in the local block
 *      (rank 0's offset is 0 by the definition of MPI_Exscan).
 *   3. Add the offset to every element in the local output block.
 *
 * On return rank r holds correct values for out[lo..hi); entries outside
 * that range are unspecified.
 */

#include <mpi.h>

#include "scan.h"

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block decomposition: rank r owns [lo, hi). */
    long lo = (long)n * rank / size;
    long hi = (long)n * (rank + 1) / size;

    /* Step 1: local inclusive scan over in[lo..hi). */
    double run = 0.0;
    for (long i = lo; i < hi; i++) {
        run += in[i];
        out[i] = run;
    }
    /* run now holds the total sum of this rank's block. */

    /* Step 2: exclusive prefix sum of block totals across ranks.
     * MPI_Exscan writes the sum of all *lower* ranks' contributions into
     * offset; rank 0 leaves offset untouched (it stays 0.0). */
    double offset = 0.0;
    MPI_Exscan(&run, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Step 3: shift the local block by the accumulated offset. */
    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
}
