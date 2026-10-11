#ifndef PROBLEM_H
#define PROBLEM_H

/* Deterministic pseudo-random fill in [0, 1): a 32-bit multiplicative hash of the
 * 0-based index, reduced modulo 2^32 and scaled. The arithmetic is exact 64-bit
 * modular arithmetic, so the C unit test, the C benchmark, and the Python oracle
 * produce bit-identical values. */
static inline double x_of(long i)
{
    unsigned long long h = ((unsigned long long)i * 2654435761ULL + 12345ULL)
                           & 0xFFFFFFFFULL;
    return (double)h / 4294967296.0;
}

/* Weighted bin checksum the oracle stores: sum over b of counts[b] * (b + 1). The
 * weight makes the checksum sensitive to WHICH bin each value landed in, not just to
 * the total count. Every term is an exact integer in double precision. */
static inline double checksum(const long *counts, int nbins)
{
    double s = 0.0;
    for (int b = 0; b < nbins; b++) s += (double)counts[b] * (double)(b + 1);
    return s;
}

#endif /* PROBLEM_H */
