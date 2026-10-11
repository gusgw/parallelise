#ifndef PROBLEM_H
#define PROBLEM_H

/* Deterministic integer fill (values are small integers, exactly representable in
 * double) so the C unit test, the C benchmark, and the Python oracle agree exactly.
 * i, j are 0-based row/column indices; matrices are n x n row-major. */
static inline double A_of(long i, long j) { return (double)(((i * 3 + j * 7) % 11) + 1); }
static inline double B_of(long i, long j) { return (double)(((i * 5 + j * 2) % 13) + 1); }

/* Sum of all entries of an n x n row-major matrix -- the checksum the oracle stores. */
static inline double checksum(const double *M, long n)
{
    double s = 0.0;
    for (long t = 0; t < n * n; t++) s += M[t];
    return s;
}

#endif /* PROBLEM_H */
