/* cg.c — OpenMP-parallel conjugate gradient for the 2D Poisson system.
 * Implements the signature declared in cg.h.
 *
 * Algorithm (from cg.ipynb / cg.h):
 *   x = 0;  r = b;  p = r;  rr = dot(r,r);
 *   for iters:
 *     q = A p                          (matrix-free 5-point stencil)
 *     alpha = rr / dot(p, q)
 *     x += alpha * p
 *     r -= alpha * q
 *     rr_new = dot(r, r)
 *     beta  = rr_new / rr
 *     rr    = rr_new
 *     p = r + beta * p
 *
 * Every vector operation (stencil, axpy, dot) is parallelised with
 * '#pragma omp parallel for'; the outer CG loop is sequential because
 * each iteration depends on the previous one.
 */

#include "cg.h"

#include <stdlib.h>   /* malloc, free */
#include <string.h>   /* memcpy       */

/* ---------- matrix-free 5-point Poisson operator: q = A*p ---------- */
static void apply_A(double * restrict q,
                    const double * restrict p,
                    long m)
{
    long n = m * m;
#pragma omp parallel for schedule(static)
    for (long idx = 0; idx < n; idx++) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * p[idx];
        if (i > 0)     s -= p[idx - m];
        if (i < m - 1) s -= p[idx + m];
        if (j > 0)     s -= p[idx - 1];
        if (j < m - 1) s -= p[idx + 1];
        q[idx] = s;
    }
}

/* ---------- dot product -------------------------------------------- */
static double dot(const double * restrict u,
                  const double * restrict v,
                  long n)
{
    double s = 0.0;
#pragma omp parallel for reduction(+:s) schedule(static)
    for (long k = 0; k < n; k++)
        s += u[k] * v[k];
    return s;
}

/* ---------- public entry point ------------------------------------- */
void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;

    double *r = (double *)malloc(sizeof(double) * (size_t)n);
    double *p = (double *)malloc(sizeof(double) * (size_t)n);
    double *q = (double *)malloc(sizeof(double) * (size_t)n);

    /* x = 0;  r = b;  p = r */
#pragma omp parallel for schedule(static)
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
    }

    double rr = dot(r, r, n);

    for (int iter = 0; iter < iters; iter++) {
        /* q = A p */
        apply_A(q, p, m);

        /* alpha = rr / dot(p, q) */
        double pq   = dot(p, q, n);
        double alpha = rr / pq;

        /* x += alpha * p;   r -= alpha * q */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }

        /* rr_new = dot(r, r) */
        double rr_new = dot(r, r, n);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++)
            p[k] = r[k] + beta * p[k];
    }

    free(r);
    free(p);
    free(q);
}
