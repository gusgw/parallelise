#ifndef DECAY_H
#define DECAY_H

/* Decaying activation score. Given the length-n stimulus array w, fill out so that
 *
 *     out[i] = max( -2.0 + out[i-1],  -1.0 + out[i-2],  w[i] )   for 0 <= i < n,
 *
 * where a missing earlier score (out[-1], out[-2]) is treated as the sentinel -1.0e9.
 * So out[0] = w[0] and out[1] = max(-2.0 + out[0], w[1]). All values are integers held
 * in double, well below 2^53, so the result is exact and must not depend on the number
 * of workers.
 *
 * Implement THIS EXACT signature in decay.c. The naive loop is inherently sequential --
 * each out[i] depends on the two scores before it -- so a serial port will be correct
 * but will not pass the benchmark's real-speedup requirement.
 * This is a plain shared-memory function: it uses no MPI. */
void decay(const double *w, double *out, long n);

#endif /* DECAY_H */
