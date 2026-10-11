#ifndef RESERVOIR_H
#define RESERVOIR_H

/* Reservoir level. Given the length-n array a of signed per-step changes, fill out so
 * that
 *
 *     out[i] = min(20000.0, max(0.0, out[i-1] + a[i]))   for 0 <= i < n,
 *
 * where the level before the first step, out[-1], is 0. So out[0] = min(20000, max(0,
 * a[0])). All values are integers held in double, in the range [0, 20000], so the result
 * is exact and must not depend on the number of ranks.
 *
 * Implement THIS EXACT signature in reservoir.c. Note this is NOT a prefix sum: the two
 * limits mean that once the level reaches 0 or the capacity, earlier changes are
 * forgotten, so summing the changes and clamping at the end gives the wrong answer.
 *
 * It is a COLLECTIVE routine: every MPI rank calls it together with identical a (each
 * rank holds a full copy). The result stays DISTRIBUTED: on return, rank r must hold
 * correct values for exactly its own contiguous block out[lo, hi), lo = n*rank/size,
 * hi = n*(rank+1)/size; entries outside the rank's own block are unspecified. Do NOT
 * gather the whole array inside this function. Assume MPI is already initialised by the
 * caller (main); do NOT call MPI_Init or MPI_Finalize.
 *
 * The naive loop is inherently sequential (each level depends on the one before), so a
 * serial port will be correct but will not pass the benchmark's real-speedup
 * requirement. To parallelise, notice that a rank's block can be processed once the level
 * entering it is known; each rank can summarise, from its own block alone, how that block
 * maps an entering level to the leaving level, and those per-block maps combine
 * associatively, so a short collective exchange of the summaries gives every rank the
 * level entering its block. Deriving the right form for that map, and how to combine two
 * of them, is the crux. */
void reservoir(const double *a, double *out, long n);

#endif /* RESERVOIR_H */
