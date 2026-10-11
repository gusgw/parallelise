#ifndef JACOBI_H
#define JACOBI_H

/* Jacobi 5-point heat stencil on an n x n row-major grid. For `iters` sweeps, replace
 * each INTERIOR cell by the average of its four neighbours, using the PREVIOUS sweep's
 * values (double-buffered):
 *     u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).
 * The four edges are FIXED boundaries and are never modified. On return u holds the
 * final grid.
 *
 * Implement THIS EXACT signature in jacobi.c. Parallelise with OpenMP (e.g. a
 * '#pragma omp parallel for' over the interior rows inside each sweep; sweeps
 * themselves are sequential because each depends on the previous one). Plain
 * shared-memory function: no MPI. Use the neighbour order above exactly so the
 * rounding matches the oracle. */
void jacobi(double *u, long n, int iters);

#endif /* JACOBI_H */
