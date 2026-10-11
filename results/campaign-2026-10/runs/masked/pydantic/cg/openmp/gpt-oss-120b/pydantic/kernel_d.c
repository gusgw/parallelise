#include "kernel_d.h"
#include "problem.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline double dot_product(const double *a, const double *b, long n)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void apply_A(const double *v, double *q, long m)
{
    long n = m * m;
    #pragma omp parallel for
    for (long idx = 0; idx < n; ++idx) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * v[idx];
        if (i > 0)   s -= v[idx - m];
        if (i < m-1) s -= v[idx + m];
        if (j > 0)   s -= v[idx - 1];
        if (j < m-1) s -= v[idx + 1];
        q[idx] = s;
    }
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    double *r = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *q = (double*)malloc(n * sizeof(double));
    if (!r || !p || !q) {
        /* allocation failure, abort */
        free(r); free(p); free(q);
        return;
    }

    // initialize x to zero
    #pragma omp parallel for
    for (long i = 0; i < n; ++i) x[i] = 0.0;

    // r = b (since x is zero), p = r
    #pragma omp parallel for
    for (long i = 0; i < n; ++i) {
        r[i] = b[i];
        p[i] = b[i];
    }

    double rr = dot_product(r, r, n);

    for (int it = 0; it < iters; ++it) {
        apply_A(p, q, m);
        double pq = dot_product(p, q, n);
        double alpha = rr / pq;

        // x += alpha * p; r -= alpha * q
        #pragma omp parallel for
        for (long i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = dot_product(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;

        // p = r + beta * p
        #pragma omp parallel for
        for (long i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
