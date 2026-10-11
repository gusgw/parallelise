#ifndef NBODY_H
#define NBODY_H

/* Direct 2D N-body acceleration step with softening. Body i has unit mass and sits at
 * (px[i], py[i]); the kernel computes, for every i, the direct sum over all j != i:
 *     ax[i] = sum of dx / d3,   ay[i] = sum of dy / d3
 * where dx = px[j] - px[i], dy = py[j] - py[i], r2 = dx*dx + dy*dy, and
 * d3 = (r2 + EPS2) * sqrt(r2 + EPS2) with EPS2 = 0.0025 (softening 0.05 squared).
 *
 * Implement THIS EXACT signature in nbody.cu. The sum runs on the GPU: the host
 * function (declared extern "C") copies px and py to the device, launches a
 * grid-stride kernel where each thread computes the accelerations of one body i
 * (looping over all j), synchronises, checks CUDA errors (cudaGetLastError /
 * cudaDeviceSynchronize), copies ax and ay back, and frees the device buffers.
 * px, py, ax and ay are host pointers. */
#ifdef __cplusplus
extern "C"
#endif
void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n);

#endif /* NBODY_H */
