#include "cg.h"
#include <stdlib.h>
#include <math.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations.
 * The operator is the 5-point stencil with Dirichlet boundaries OUTSIDE the grid.
 */

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));

    /* Initialize x = 0, r = b, p = r */
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = r[k];
    }

    /* rr = dot(r, r) */
    double rr = 0.0;
    #pragma omp parallel for reduction(+:rr)
    for (long k = 0; k < n; k++) {
        rr += r[k] * r[k];
    }

    for (int iter = 0; iter < iters; iter++) {
        /* q = A * p (5-point stencil) */
        #pragma omp parallel for
        for (long i = 0; i < m; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                double s = 4.0 * p[idx];
                if (i > 0)     s -= p[idx - m];
                if (i < m - 1) s -= p[idx + m];
                if (j > 0)     s -= p[idx - 1];
                if (j < m - 1) s -= p[idx + 1];
                q[idx] = s;
            }
        }

        /* pdotq = dot(p, q) */
        double pdotq = 0.0;
        #pragma omp parallel for reduction(+:pdotq)
        for (long k = 0; k < n; k++) {
            pdotq += p[k] * q[k];
        }

        double alpha = rr / pdotq;

        /* x += alpha * p */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
        }

        /* r -= alpha * q */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            r[k] -= alpha * q[k];
        }

        /* rr_new = dot(r, r) */
        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (long k = 0; k < n; k++) {
            rr_new += r[k] * r[k];
        }

        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            p[k] = r[k] + beta * p[k];
        }
    }

    free(r);
    free(p);
    free(q);
}
