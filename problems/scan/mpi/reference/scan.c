#include <mpi.h>
#include "scan.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Block-scan
 * decomposition: each rank scans its contiguous block [lo, hi) of in locally, an
 * MPI_Exscan of the block totals gives the rank's starting offset (0 on rank 0), and
 * the offset is added to the block. The result stays distributed per the contract in
 * scan.h. Collective; assumes MPI already initialised. */
void scan(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;

    /* local inclusive scan of the owned block; total is the block's sum */
    double total = 0.0;
    for (long i = lo; i < hi; i++) {
        total += in[i];
        out[i] = total;
    }

    /* exclusive scan of the block totals -> this rank's starting offset. MPI_Exscan
     * leaves rank 0's receive buffer undefined, so force its offset to zero. */
    double offset = 0.0;
    MPI_Exscan(&total, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) offset = 0.0;
    for (long i = lo; i < hi; i++) out[i] += offset;
    /* the result stays distributed: this rank's block of out is now correct, and that
     * is the contract -- no gather here (it would cost as much as the scan itself) */
}
