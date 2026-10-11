#ifndef PROBLEM_H
#define PROBLEM_H

/* Initial + boundary field for the heat plate: the top row (i == 0) is hot (1.0) and
 * everything else starts at 0.0. The four edges are FIXED boundaries (never updated by
 * the stencil); the interior evolves. i, j are 0-based; the grid is n x n row-major. */
static inline double u_init(long i, long j, long n)
{
    (void)j; (void)n;
    return (i == 0) ? 1.0 : 0.0;
}

/* Sum of all cells -- the checksum the oracle stores. */
static inline double checksum(const double *u, long n)
{
    double s = 0.0;
    for (long t = 0; t < n * n; t++) s += u[t];
    return s;
}

#endif /* PROBLEM_H */
