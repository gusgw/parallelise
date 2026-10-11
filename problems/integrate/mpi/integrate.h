#ifndef INTEGRATE_H
#define INTEGRATE_H

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals; on [0, 1] this converges to pi.
 *
 * Implement THIS EXACT signature in integrate.c. It is a COLLECTIVE routine:
 * every MPI rank calls it together and it returns the SAME full integral on
 * every rank. Assume MPI is already initialised by the caller (main); do NOT
 * call MPI_Init or MPI_Finalize inside it. Split the n subintervals across the
 * ranks (each rank sums only its own block) and combine the partial sums with
 * MPI_Allreduce. */
double integrate(double a, double b, long n);

#endif /* INTEGRATE_H */
