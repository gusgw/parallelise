#include "integrate.h"
#include <mpi.h>

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals; on [0, 1] this converges to pi.
 *
 * This is a COLLECTIVE routine: every MPI rank calls it together and it
 * returns the SAME full integral on every rank. Split the n subintervals
 * across the ranks (each rank sums only its own block) and combine the
 * partial sums with MPI_Allreduce. */
double integrate(double a, double b, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Decompose the work: each rank computes its own block of iterations */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    
    double h = (b - a) / n;
    double local_sum = 0.0;
    
    /* Each rank computes only its share of the sum */
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    
    /* Combine partial sums from all ranks */
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return global_sum * h;
}
