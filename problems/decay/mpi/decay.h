#ifndef DECAY_H
#define DECAY_H

/* Decaying activation score. Given the length-n stimulus array w, fill out so that
 *
 *     out[i] = max( -2.0 + out[i-1],  -1.0 + out[i-2],  w[i] )   for 0 <= i < n,
 *
 * where a missing earlier score (out[-1], out[-2]) is treated as the sentinel -1.0e9.
 * So out[0] = w[0] and out[1] = max(-2.0 + out[0], w[1]). All values are integers held
 * in double, well below 2^53, so the result is exact and must not depend on the number
 * of ranks.
 *
 * Implement THIS EXACT signature in decay.c. It is a COLLECTIVE routine: every MPI rank
 * calls it together with identical w (each rank holds a full copy). The result stays
 * DISTRIBUTED: on return, rank r must hold correct values for exactly its own contiguous
 * block out[lo, hi), lo = n*rank/size, hi = n*(rank+1)/size; entries outside the rank's
 * own block are unspecified. Do NOT gather the whole array inside this function. Assume
 * MPI is already initialised by the caller (main); do NOT call MPI_Init or MPI_Finalize.
 *
 * The naive loop is inherently sequential (each out[i] depends on the two scores before
 * it), so a serial port will be correct but will not pass the benchmark's real-speedup
 * requirement. */
void decay(const double *w, double *out, long n);

#endif /* DECAY_H */
