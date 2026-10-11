#ifndef CG_H
#define CG_H

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations (no convergence
 * test -- the run must be deterministic). The operator is the 5-point stencil with
 * Dirichlet boundaries OUTSIDE the grid, so a neighbour off the edge contributes zero:
 *     (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1].
 * Standard CG from x = 0:  r = b; p = r; rr = dot(r,r);  then `iters` times:
 *     q = A p;  alpha = rr / dot(p,q);  x += alpha p;  r -= alpha q;
 *     rr_new = dot(r,r);  beta = rr_new / rr;  rr = rr_new;  p = r + beta p.
 * On return x holds the iterate after `iters` iterations.
 *
 * Implement THIS EXACT signature in cg.cu. The solve runs on the GPU: the host
 * function (declared extern "C") copies b to the device, keeps device buffers for x,
 * r, p and q, and drives the iterations from the host. One 2D-grid kernel applies the
 * operator (q = A p), simple grid-stride kernels do the axpy updates, and the two dot
 * products use a per-block shared-memory reduction kernel whose block partials are
 * cudaMemcpy'd back and summed on the host (keep it simple; correctness first).
 * Synchronise and check CUDA errors (cudaGetLastError / cudaDeviceSynchronize), copy
 * the final x back to the host, and free the device buffers. x and b are host
 * pointers. */
#ifdef __cplusplus
extern "C"
#endif
void cgsolve(double *x, const double *b, long m, int iters);

#endif /* CG_H */
