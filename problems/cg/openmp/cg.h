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
 * Implement THIS EXACT signature in cg.c. Parallelise with OpenMP INSIDE each vector
 * operation (e.g. '#pragma omp parallel for' over the grid for q = A p and for the
 * axpy updates, with 'reduction(+:...)' for the two dot products); the iterations
 * themselves are sequential because each depends on the previous one. Plain
 * shared-memory function: no MPI. */
void cgsolve(double *x, const double *b, long m, int iters);

#endif /* CG_H */
