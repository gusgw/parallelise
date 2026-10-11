/* cg.c — OpenMP-parallel conjugate gradient for the 2-D Poisson system.
 *
 * Implements the signature declared in cg.h.  The CG outer loop is sequential
 * (each iteration depends on the previous one); every inner vector operation
 * (stencil application, axpy updates, dot products) is parallelised with
 * '#pragma omp parallel for'.
 */

#include "cg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * apply_A  –  matrix-free 5-point Poisson stencil
 *   (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1]
 * Neighbours outside the m×m grid contribute zero (Dirichlet BCs).
 * --------------------------------------------------------------------------- */
static void apply_A(double *restrict q,
                    const double *restrict v,
                    long m)
{
    long n = m * m;

#pragma omp parallel for schedule(static)
    for (long idx = 0; idx < n; idx++) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * v[idx];
        if (i > 0)     s -= v[idx - m];
        if (i < m - 1) s -= v[idx + m];
        if (j > 0)     s -= v[idx - 1];
        if (j < m - 1) s -= v[idx + 1];
        q[idx] = s;
    }
}

/* ---------------------------------------------------------------------------
 * cgsolve  –  standard CG from x = 0, run for exactly `iters` iterations.
 * --------------------------------------------------------------------------- */
void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;

    /* Allocate work vectors. */
    double *r = (double *)malloc(sizeof(double) * (size_t)n);
    double *p = (double *)malloc(sizeof(double) * (size_t)n);
    double *q = (double *)malloc(sizeof(double) * (size_t)n);

    /* x = 0 */
#pragma omp parallel for schedule(static)
    for (long k = 0; k < n; k++) x[k] = 0.0;

    /* r = b;  p = r */
    memcpy(r, b, sizeof(double) * (size_t)n);
    memcpy(p, b, sizeof(double) * (size_t)n);

    /* rr = dot(r, r) */
    double rr = 0.0;
#pragma omp parallel for reduction(+:rr) schedule(static)
    for (long k = 0; k < n; k++) rr += r[k] * r[k];

    for (int iter = 0; iter < iters; iter++) {
        /* q = A p */
        apply_A(q, p, m);

        /* alpha = rr / dot(p, q) */
        double pq = 0.0;
#pragma omp parallel for reduction(+:pq) schedule(static)
        for (long k = 0; k < n; k++) pq += p[k] * q[k];
        double alpha = rr / pq;

        /* x += alpha * p;   r -= alpha * q */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }

        /* rr_new = dot(r, r) */
        double rr_new = 0.0;
#pragma omp parallel for reduction(+:rr_new) schedule(static)
        for (long k = 0; k < n; k++) rr_new += r[k] * r[k];

        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++) p[k] = r[k] + beta * p[k];
    }

    free(r);
    free(p);
    free(q);
}
