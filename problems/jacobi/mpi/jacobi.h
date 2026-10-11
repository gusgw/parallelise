#ifndef JACOBI_H
#define JACOBI_H

/* Jacobi 5-point heat stencil on an n x n row-major grid. For `iters` sweeps, replace
 * each INTERIOR cell by the average of its four neighbours, using the PREVIOUS sweep's
 * values (double-buffered):
 *     u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).
 * The four edges are FIXED boundaries and are never modified.
 *
 * Implement THIS EXACT signature in jacobi.c. It is a COLLECTIVE routine: every MPI rank
 * calls it together with an identical initial grid, and on return every rank holds the
 * full final grid u. Assume MPI is already initialised by the caller (main); do NOT call
 * MPI_Init or MPI_Finalize inside it. Decompose the grid rows across the ranks; each
 * sweep, exchange the boundary ("halo") rows with the neighbouring ranks (MPI_Sendrecv)
 * so every rank can update its own rows from the previous sweep; assemble the full grid
 * on every rank at the end. Use the neighbour order above exactly so the rounding matches
 * the oracle. */
void jacobi(double *u, long n, int iters);

#endif /* JACOBI_H */
