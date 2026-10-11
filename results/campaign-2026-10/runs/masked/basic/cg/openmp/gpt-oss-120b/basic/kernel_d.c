#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kernel_d.h"

/* Apply the 5-point stencil matrix A to vector v (size m*m).
 * Result is stored in q (pre-allocated).
 */
static void apply_A(const double *v, double *q, long m)
{
    long n = m * m;
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < n; ++idx) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * v[idx];
        if (i > 0)      s -= v[idx - m];
        if (i < m - 1)  s -= v[idx + m];
        if (j > 0)      s -= v[idx - 1];
        if (j < m - 1)  s -= v[idx + 1];
        q[idx] = s;
    }
}

static double dot(const double *u, const double *v, long n)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < n; ++i) {
        sum += u[i] * v[i];
    }
    return sum;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    if (!r || !p || !q) {
        /* Allocation failure: abort silently (caller may check x) */
        free(r); free(p); free(q);
        return;
    }

    /* Initialize x to zero, r = b, p = r */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    double rr = dot(r, r, n);

    for (int it = 0; it < iters; ++it) {
        apply_A(p, q, m);
        double pq = dot(p, q, n);
        double alpha = rr / pq;

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
