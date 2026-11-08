/* ising: generated from a notebook's declaration by parallelise. Do not change this file.
 *
 * Write the routine declared below so that it gives the same results as the notebook's
 * function for any arguments of the declared shapes. Every process receives all the
 * arguments. When the routine returns, every result must be complete in the caller's
 * memory: for MPI, on process 0; for CUDA, in ordinary memory, copied back from the
 * graphics card. The routine must compute its results from its arguments alone: it may keep
 * nothing from one call to the next, read no file, no clock and no environment variable,
 * and start no process or thread except through its backend. It may use the C library's
 * mathematics, memory and printing functions and the calls of its backend.
 */
#ifndef ISING_H
#define ISING_H

#ifdef __cplusplus
extern "C" {
#endif

void ising(long side, double beta, long equilibration, long sweeps, long seed, double *energy, double *magnetisation);

#ifdef __cplusplus
}
#endif

#endif
