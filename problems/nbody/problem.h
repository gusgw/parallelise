#ifndef PROBLEM_H
#define PROBLEM_H

#include <math.h>

/* Deterministic body positions in [-1, 1): exact integer arithmetic followed by the
 * same division, multiply and subtract in the same order as the Python, so the C unit
 * test, the C benchmark, and the Python oracle agree bit for bit. i is the 0-based
 * body index. */
static inline double px_of(long i) { return (double)((i * 73L) % 4093L) / 4093.0 * 2.0 - 1.0; }
static inline double py_of(long i) { return (double)((i * 151L) % 4093L) / 4093.0 * 2.0 - 1.0; }

/* Sum over the bodies of |ax_i| + |ay_i| -- the checksum the oracle stores. The
 * absolute values avoid the near-total cancellation that makes the raw sum of the
 * accelerations ~0. */
static inline double checksum(const double *ax, const double *ay, long n)
{
    double s = 0.0;
    for (long i = 0; i < n; i++) s += fabs(ax[i]) + fabs(ay[i]);
    return s;
}

#endif /* PROBLEM_H */
