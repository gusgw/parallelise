#ifndef PROBLEM_H
#define PROBLEM_H

/* Shared inputs for the decaying-activation-score problem, so the C unit test, the C
 * benchmark, and the Python oracle agree exactly.
 *
 * The recurrence per step is
 *     out[i] = max(A + out[i-1], B + out[i-2], w[i]),
 * with A = -2 and B = -1, and out[i-1], out[i-2] taken as the sentinel NEG for the two
 * steps that have no history (so out[0] = w[0] and out[1] = max(A + out[0], w[1])).
 * Every quantity is an integer that stays well below 2^53, so double precision is exact
 * and the answer does not depend on how the work is split. */

#define DECAY_A (-2.0)
#define DECAY_B (-1.0)
#define DECAY_NEG (-1.0e9)   /* stands in for a missing earlier score */

/* Deterministic stimulus in the integer range [0, 1000). The hash uses unsigned 64-bit
 * arithmetic (i * 2654435761 stays far below 2^63 for every index used here), so C and
 * Python compute the same integer, and the (double) result is exact. i is 0-based. */
static inline double stim_of(long i)
{
    unsigned long long h = (unsigned long long)i * 2654435761ULL % 1000003ULL % 1000ULL;
    return (double)h;
}

/* Sum of all entries of the length-n output -- the checksum the oracle stores. */
static inline double checksum(const double *out, long n)
{
    double s = 0.0;
    for (long t = 0; t < n; t++) s += out[t];
    return s;
}

#endif /* PROBLEM_H */
