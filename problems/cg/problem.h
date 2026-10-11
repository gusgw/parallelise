#ifndef PROBLEM_H
#define PROBLEM_H

/* Deterministic right-hand side for the Poisson solve (values in [0, 1), computed from
 * exact small-integer arithmetic) so the C unit test, the C benchmark, and the Python
 * oracle all solve the identical system. idx is the row-major index i * m + j into the
 * m x m grid. */
static inline double b_of(long idx) { return (double)((idx * 97) % 211) / 211.0; }

/* Sum of all entries of the length-m*m solution vector -- the checksum the oracle stores. */
static inline double checksum(const double *x, long m)
{
    double s = 0.0;
    for (long t = 0; t < m * m; t++) s += x[t];
    return s;
}

#endif /* PROBLEM_H */
