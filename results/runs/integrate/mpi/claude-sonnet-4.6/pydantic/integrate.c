/* integrate.c — parallel midpoint-rule estimate of ∫ 4/(1+x²) dx over [a,b].
 *
 * The n subintervals are split evenly across MPI ranks (block decomposition).
 * Each rank accumulates its partial sum; MPI_Allreduce combines them so every
 * rank returns the same full integral.  MPI must already be initialised by the
 * caller; this routine does NOT call MPI_Init / MPI_Finalize.
 */

#include "integrate.h"

#include <mpi.h>

double integrate(double a, double b, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block decomposition: rank r owns subintervals [lo, hi). */
    long lo = (long)n * rank / size;
    long hi = (long)n * (rank + 1) / size;

    double h = (b - a) / (double)n;

    /* Accumulate the partial sum for this rank's slice. */
    double local_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    local_sum *= h;

    /* Combine partial sums: every rank receives the full result. */
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    return global_sum;
}
