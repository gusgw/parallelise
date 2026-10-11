#ifndef PROBLEM_H
#define PROBLEM_H

/* Shared inputs for the reservoir-level problem, so the C unit test, the C benchmark, and
 * the Python oracle agree exactly.
 *
 * The recurrence per step is
 *     out[i] = min(CAP, max(0, out[i-1] + a[i])),
 * with out[-1] taken as 0. Every quantity is an integer that stays well below 2^53, so
 * double precision is exact and the answer does not depend on how the work is split. */

#define RES_CAP (20000.0)   /* tank capacity; the level is held to [0, CAP] */

/* Deterministic signed change per step in the integer range [-500, 500]. The hash uses
 * unsigned 64-bit arithmetic (i * 2654435761 stays far below 2^63 for every index used
 * here), so C and Python compute the same integer, and the (double) result is exact.
 * i is 0-based. */
static inline double change_of(long i)
{
    unsigned long long h = (unsigned long long)i * 2654435761ULL % 1000003ULL % 1001ULL;
    return (double)h - 500.0;
}

/* Sum of all entries of the length-n output -- the checksum the oracle stores. */
static inline double checksum(const double *out, long n)
{
    double s = 0.0;
    for (long t = 0; t < n; t++) s += out[t];
    return s;
}

#endif /* PROBLEM_H */
