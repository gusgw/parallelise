#ifndef JACOBI_H
#define JACOBI_H

/* Jacobi 5-point heat stencil on an n x n row-major grid. For `iters` sweeps, replace
 * each INTERIOR cell by the average of its four neighbours, using the PREVIOUS sweep's
 * values (double-buffered):
 *     u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).
 * The four edges are FIXED boundaries and are never modified. On return u holds the
 * final grid.
 *
 * Implement THIS EXACT signature in jacobi.cu. The sweeps run on the GPU: the host
 * function (declared extern "C") copies u to a device buffer, keeps TWO device buffers
 * and alternates them (kernel reads one, writes the interior of the other; boundaries
 * are copied once), launches one 2D-grid kernel per sweep, synchronises, checks CUDA
 * errors (cudaGetLastError / cudaDeviceSynchronize), copies the final grid back to u,
 * and frees the buffers. Use the neighbour order above exactly. */
#ifdef __cplusplus
extern "C"
#endif
void jacobi(double *u, long n, int iters);

#endif /* JACOBI_H */
