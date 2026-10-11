/* scan.c — parallel inclusive prefix sum using MPI block-scan decomposition.
 *
 * Strategy (matches scan.h specification):
 *   1. Each rank computes its block bounds: lo = n*rank/size, hi = n*(rank+1)/size.
 *   2. Each rank performs a local inclusive scan over in[lo..hi) into out[lo..hi).
 *   3. MPI_Exscan(MPI_SUM) of each rank's block total gives the exclusive prefix
 *      offset (rank 0's offset is defined as 0 since Exscan leaves it undefined).
 *   4. Each rank adds the offset to every element in its local output block.
 *
 * On return, rank r holds correct values for out[lo..hi); entries outside that
 * range are unspecified (no full gather is performed). */

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

    /* Step 1: local inclusive scan over this rank's slice. */
    double run = 0.0;
    for (long i = lo; i < hi; i++) {
        run += in[i];
        out[i] = run;
    }
    /* run now holds the total sum of this rank's block. */

    /* Step 2: exclusive prefix sum of block totals across ranks.
     * MPI_Exscan gives rank r the sum of block totals for ranks 0..r-1.
     * The result for rank 0 is undefined by the MPI standard, so we zero it. */
    double offset = 0.0;
    MPI_Exscan(&run, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) offset = 0.0;

    /* Step 3: shift local output by the incoming offset. */
    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
}
