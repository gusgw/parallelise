#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cg.h"

/* Helper: dot product with OpenMP reduction */
static double dot_product(const double *a, const double *b, long n)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Helper: apply 5-point Poisson operator A to vector src, store in dst */
static void apply_A(const double *src, double *dst, long m)
{
    long n = m * m;
    #pragma omp parallel for
    for (long idx = 0; idx < n; ++idx) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * src[idx];
        if (i > 0)   s -= src[idx - m];      // up
        if (i < m-1) s -= src[idx + m];      // down
        if (j > 0)   s -= src[idx - 1];      // left
        if (j < m-1) s -= src[idx + 1];      // right
        dst[idx] = s;
    }
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    /* Allocate temporary vectors */
    double *r = (double *)malloc(sizeof(double) * n);
    double *p = (double *)malloc(sizeof(double) * n);
    double *q = (double *)malloc(sizeof(double) * n);
    if (!r || !p || !q) {
        /* Allocation failure: abort */
        free(r); free(p); free(q);
        return;
    }

    /* Initialise x to zero */
    #pragma omp parallel for
    for (long i = 0; i < n; ++i) x[i] = 0.0;

    /* r = b, p = r */
    memcpy(r, b, sizeof(double) * n);
    memcpy(p, b, sizeof(double) * n);

    double rr = dot_product(r, r, n);

    for (int it = 0; it < iters; ++it) {
        apply_A(p, q, m);
        double pq = dot_product(p, q, n);
        double alpha = rr / pq;

        /* x += alpha * p */
        #pragma omp parallel for
        for (long i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
        }
        /* r -= alpha * q */
        #pragma omp parallel for
        for (long i = 0; i < n; ++i) {
            r[i] -= alpha * q[i];
        }
        double rr_new = dot_product(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        /* p = r + beta * p */
        #pragma omp parallel for
        for (long i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
