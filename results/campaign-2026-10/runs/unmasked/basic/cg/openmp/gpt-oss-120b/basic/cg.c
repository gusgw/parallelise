#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cg.h"

/* Helper: apply 5-point stencil A to vector src, store in dst.
 * Both src and dst are length m*m arrays.
 */
static void apply_A(const double *src, double *dst, long m)
{
    long n = m * m;
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < n; idx++) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * src[idx];
        if (i > 0)     s -= src[idx - m];
        if (i < m - 1) s -= src[idx + m];
        if (j > 0)     s -= src[idx - 1];
        if (j < m - 1) s -= src[idx + 1];
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
        /* Allocation failure: clean up and return */
        free(r); free(p); free(q);
        return;
    }

    /* Initialize x to zero */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) x[i] = 0.0;

    /* r = b, p = r */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = b[i];
    }

    /* rr = dot(r,r) */
    double rr = 0.0;
    #pragma omp parallel for reduction(+:rr) schedule(static)
    for (long i = 0; i < n; i++) rr += r[i] * r[i];

    for (int it = 0; it < iters; it++) {
        /* q = A * p */
        apply_A(p, q, m);

        /* dot(p,q) */
        double dotpq = 0.0;
        #pragma omp parallel for reduction(+:dotpq) schedule(static)
        for (long i = 0; i < n; i++) dotpq += p[i] * q[i];

        double alpha = rr / dotpq;

        /* x += alpha * p */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) x[i] += alpha * p[i];

        /* r -= alpha * q */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) r[i] -= alpha * q[i];

        /* rr_new = dot(r,r) */
        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new) schedule(static)
        for (long i = 0; i < n; i++) rr_new += r[i] * r[i];

        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    }

    free(r);
    free(p);
    free(q);
}
