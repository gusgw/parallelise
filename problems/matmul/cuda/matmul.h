#ifndef MATMUL_H
#define MATMUL_H

/* Dense matrix multiply C = A * B for n x n row-major matrices.
 *
 * Implement THIS EXACT signature in matmul.cu. The multiply runs on the GPU: the
 * host function (declared extern "C") copies A and B to the device, launches a 2D
 * grid of threads where each thread computes one entry (or a small tile) of C,
 * synchronises, checks CUDA errors (cudaGetLastError / cudaDeviceSynchronize),
 * copies C back, and frees the device buffers. A, B and C are host pointers. */
#ifdef __cplusplus
extern "C"
#endif
void matmul(const double *A, const double *B, double *C, long n);

#endif /* MATMUL_H */
