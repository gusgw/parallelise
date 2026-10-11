#ifndef PROBLEM_H
#define PROBLEM_H

/* Deterministic pseudo-random fill in [-0.5, 0.5) so the C unit test, the C benchmark,
 * and the Python oracle agree exactly. The hash uses unsigned 64-bit arithmetic
 * (i * 2654435761 stays far below 2^63 for every index used here), so C and Python
 * compute the same integer and the same double, bit for bit. i is the 0-based index. */
static inline double in_of(long i)
{
    unsigned long long h = ((unsigned long long)i * 2654435761ULL) % 1000003ULL;
    return (double)h / 1000003.0 - 0.5;
}

/* Sum of all entries of the length-n output -- the checksum the oracle stores. */
static inline double checksum(const double *out, long n)
{
    double s = 0.0;
    for (long t = 0; t < n; t++) s += out[t];
    return s;
}

#endif /* PROBLEM_H */
