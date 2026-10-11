#ifndef RESERVOIR_H
#define RESERVOIR_H

/* Reservoir level. Given the length-n array a of signed per-step changes, fill out so
 * that
 *
 *     out[i] = min(20000.0, max(0.0, out[i-1] + a[i]))   for 0 <= i < n,
 *
 * where the level before the first step, out[-1], is 0. So out[0] = min(20000, max(0,
 * a[0])). All values are integers held in double, in the range [0, 20000], so the result
 * is exact and must not depend on the number of workers.
 *
 * Implement THIS EXACT signature in reservoir.c. Note this is NOT a prefix sum: the two
 * limits mean that once the level reaches 0 or the capacity, earlier changes are
 * forgotten, so summing the changes and clamping at the end gives the wrong answer.
 *
 * The naive loop is inherently sequential (each level depends on the one before), so a
 * serial port will be correct but will not pass the benchmark's real-speedup
 * requirement. To parallelise, notice that a block of consecutive steps can be processed
 * independently of the actual level entering it if you can express, once per block, how
 * that block maps an entering level to the leaving level; those per-block maps combine
 * associatively, so a two-pass block scheme works. Deriving the right form for that map,
 * and how to combine two of them, is the crux. Plain shared-memory function: no MPI. */
void reservoir(const double *a, double *out, long n);

#endif /* RESERVOIR_H */
