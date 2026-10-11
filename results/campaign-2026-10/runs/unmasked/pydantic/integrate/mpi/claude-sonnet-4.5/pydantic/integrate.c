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
    
    /* Divide the n subintervals across ranks */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    
    /* Compute the step size */
    double h = (b - a) / n;
    
    /* Each rank computes its partial sum */
    double partial_sum = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        partial_sum += 4.0 / (1.0 + x * x);
    }
    partial_sum *= h;
    
    /* Combine all partial sums with MPI_Allreduce */
    double total_sum = 0.0;
    MPI_Allreduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return total_sum;
}
