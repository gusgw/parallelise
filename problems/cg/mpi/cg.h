#ifndef CG_H
#define CG_H

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations (no convergence
 * test -- the run must be deterministic and every rank must execute the same
 * collectives the same number of times). The operator is the 5-point stencil with
 * Dirichlet boundaries OUTSIDE the grid, so a neighbour off the edge contributes zero:
 *     (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1].
 * Standard CG from x = 0:  r = b; p = r; rr = dot(r,r);  then `iters` times:
 *     q = A p;  alpha = rr / dot(p,q);  x += alpha p;  r -= alpha q;
 *     rr_new = dot(r,r);  beta = rr_new / rr;  rr = rr_new;  p = r + beta p.
 * On return x holds the iterate after `iters` iterations.
 *
 * Implement THIS EXACT signature in cg.c. It is a COLLECTIVE routine: every MPI rank
 * calls it together with identical b (each rank holds a full copy), and on return
 * every rank holds the full solution x. Assume MPI is already initialised by the
 * caller (main); do NOT call MPI_Init or MPI_Finalize inside it. Decompose the grid
 * rows across the ranks: each rank owns rows [lo, hi) of x, r, p and q and updates
 * only those. Before EACH application of A, exchange one boundary row of p up and one
 * down with the neighbouring ranks (MPI_Sendrecv, with MPI_PROC_NULL at the edges so
 * the ends need no special cases). Compute the two dot products from local partial
 * sums combined with MPI_Allreduce (every rank needs alpha and beta). Assemble the
 * full x on every rank with a single MPI_Allgatherv at the end. */
void cgsolve(double *x, const double *b, long m, int iters);

#endif /* CG_H */
