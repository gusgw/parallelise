#include "kernel_a.h"
#include <mpi.h>
#include <math.h>

/* Parallel implementation of the midpoint rule integration for
 *   \int_a^b 4/(1+x^2) dx
 * The work is divided evenly among MPI ranks. Each rank computes a partial
 * sum of the integrand values (without the final *h factor). The partial
 * sums are reduced with MPI_Allreduce to obtain the total sum, which is then
 * multiplied by the step size h.
 */

double kernel_a(double a, double b, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double h = (b - a) / (double)n;
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double local_sum = 0.0;
    for (long i = lo; i < hi; ++i) {
        double x = a + ( (double)i + 0.5 ) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double total_sum = 0.0;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return total_sum * h;
}
