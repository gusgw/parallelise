#ifndef INTEGRATE_H
#define INTEGRATE_H

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals. On [0, 1] this converges to pi. The implementation parallelises
 * the subinterval sum with an OpenMP reduction. */
double integrate(double a, double b, long n);

#endif /* INTEGRATE_H */
