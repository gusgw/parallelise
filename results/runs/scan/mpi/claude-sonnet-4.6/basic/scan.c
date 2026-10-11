/* scan.c — parallel inclusive prefix sum (MPI collective).
 *
 * Algorithm (block-scan decomposition, as specified in scan.h):
 *   1. Each rank owns the contiguous slice [lo, hi) of the output,
 *      where lo = n*rank/size, hi = n*(rank+1)/size.
 *   2. Each rank performs a local inclusive scan over in[lo..hi).
 *   3. MPI_Exscan(MPI_SUM) propagates the block totals so every rank
 *      receives the sum of all preceding ranks' blocks as its offset
 *      (rank 0's offset is 0 by the definition of Exscan).
 *   4. Each rank adds the offset to every element of its local block.
 *
 * MPI is already initialised by the caller; we do NOT call MPI_Init /
 * MPI_Finalize here.
 */

#include "scan.h"

#include <mpi.h>
#include <stdlib.h>   /* (nothing from stdlib needed now, but keep for safety) */

void scan(const double *in, double *out, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- 1. Determine this rank's slice [lo, hi) ---- */
    long lo = (long)n *  rank      / size;
    long hi = (long)n * (rank + 1) / size;

    /* ---- 2. Local inclusive scan over in[lo..hi) ---- */
    double running = 0.0;
    for (long i = lo; i < hi; i++) {
        running += in[i];
        out[i]   = running;
    }
    /* 'running' now holds the total sum of this rank's block. */

    /* ---- 3. Exclusive scan of block totals across ranks ---- */
    /* MPI_Exscan: rank r receives the sum of block totals from ranks 0..r-1.
     * Rank 0's receive buffer is left undefined by the standard, so we
     * initialise offset to 0.0 before the call; rank 0 keeps that value. */
    double offset = 0.0;
    MPI_Exscan(&running, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* ---- 4. Add the offset to every element in the local block ---- */
    /* Rank 0: offset stays 0.0 (Exscan leaves it untouched for rank 0),
     * so the loop is a no-op for rank 0 — correct and branch-free. */
    if (offset != 0.0) {
        for (long i = lo; i < hi; i++) {
            out[i] += offset;
        }
    }
}
