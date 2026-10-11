#include <mpi.h>
#include <stdlib.h>

#include "scan.h"

void scan(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute local block range: [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Local inclusive scan on the block */
    double run = 0.0;
    for (long i = 0; i < local_n; i++) {
        run += in[lo + i];
        out[lo + i] = run;
    }

    /* Total sum of this rank's block */
    double local_total = run;

    /* Get the sum of all lower ranks' totals using MPI_Scan
     * MPI_Scan computes an inclusive prefix sum across ranks.
     * We want the sum of all *previous* ranks, so we use MPI_Exscan
     * which gives an exclusive prefix sum (sum of all ranks < current).
     * For rank 0, the result is 0 (no lower ranks).
     */
    double offset = 0.0;
    MPI_Exscan(&local_total, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* Note: MPI_Exscan leaves the receive buffer undefined on rank 0,
     * but we initialized offset = 0.0, which is correct for rank 0.
     */

    /* Add offset to each element in the local block */
    for (long i = 0; i < local_n; i++) {
        out[lo + i] += offset;
    }
}
