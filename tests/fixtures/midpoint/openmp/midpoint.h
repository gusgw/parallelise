#ifndef MIDPOINT_H
#define MIDPOINT_H

/* Midpoint-rule estimate of the integral of sin(x) over [a, b] using n
 * subintervals. The sum is parallelised with an OpenMP reduction. */
double midpoint(double a, double b, long n);

#endif /* MIDPOINT_H */
