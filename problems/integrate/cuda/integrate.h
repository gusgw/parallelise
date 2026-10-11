#ifndef INTEGRATE_H
#define INTEGRATE_H

/* Midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n
 * subintervals; on [0, 1] this converges to pi.
 *
 * Implement THIS EXACT signature in integrate.cu. The sum runs on the GPU: a
 * grid-stride CUDA kernel where each thread accumulates its share of the
 * subintervals, a per-block shared-memory reduction, and a final combine of the
 * per-block partial sums. The function is a normal host call: it launches the
 * kernel, waits, and returns the finished integral. Check CUDA API and kernel
 * errors (cudaGetLastError / cudaDeviceSynchronize). */
#ifdef __cplusplus
extern "C"
#endif
double integrate(double a, double b, long n);

#endif /* INTEGRATE_H */
