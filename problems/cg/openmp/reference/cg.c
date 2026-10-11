#include <stdlib.h>

#include "cg.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Each vector
 * operation of every CG iteration parallelises over the grid; the two dot products use
 * OpenMP reductions. The updates of x and r fuse into one loop with the dot of the new
 * residual (fusing only reorders the sums, which the oracle's tolerance allows). The
 * iterations themselves are sequential because each depends on the previous one. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    const long n = m * m;
    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);

    double rr = 0.0;
    #pragma omp parallel for reduction(+:rr)
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
        rr += b[k] * b[k];
    }

    for (int it = 0; it < iters; it++) {
        /* q = A p for the 5-point operator; neighbours off the edge are zero */
        double pq = 0.0;
        #pragma omp parallel for reduction(+:pq)
        for (long i = 0; i < m; i++)
            for (long j = 0; j < m; j++) {
                const long idx = i * m + j;
                double s = 4.0 * p[idx];
                if (i > 0)     s -= p[idx - m];
                if (i < m - 1) s -= p[idx + m];
                if (j > 0)     s -= p[idx - 1];
                if (j < m - 1) s -= p[idx + 1];
                q[idx] = s;
                pq += p[idx] * s;
            }

        const double alpha = rr / pq;
        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
            rr_new += r[k] * r[k];
        }

        const double beta = rr_new / rr;
        rr = rr_new;
        #pragma omp parallel for
        for (long k = 0; k < n; k++)
            p[k] = r[k] + beta * p[k];
    }

    free(r);
    free(p);
    free(q);
}
