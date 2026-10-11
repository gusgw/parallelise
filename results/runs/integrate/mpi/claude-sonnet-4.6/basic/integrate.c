/* integrate.c — parallel midpoint-rule estimate of ∫ 4/(1+x²) dx over [a,b].
 *
 * On [0,1] the integral equals π.  The n subintervals are split evenly across
 * MPI ranks; each rank accumulates its partial sum and MPI_Allreduce combines
 * them so every rank returns the same full result.
 *
 * Collective: MPI must already be initialised by the caller.  Do NOT call
 * MPI_Init / MPI_Finalize here.
 */

#include "integrate.h"

#include <mpi.h>

double integrate(double a, double b, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block-decompose the n subintervals across ranks.
     * rank r owns indices [lo, hi).  The ceiling-free formula
     *   lo = n * rank / size,  hi = n * (rank+1) / size
     * distributes every interval to exactly one rank with no overlap. */
    long lo = (long)rank       * n / (long)size;
    long hi = (long)(rank + 1) * n / (long)size;

    double h = (b - a) / (double)n;

    /* Accumulate the partial sum for this rank's slice. */
    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + ((double)i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    local_sum *= h;

    /* Combine partial sums: every rank receives the full integral. */
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    return global_sum;
}
