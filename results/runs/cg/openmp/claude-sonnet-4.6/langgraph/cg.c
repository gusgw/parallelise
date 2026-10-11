/* cg.c — OpenMP-parallel conjugate gradient for the 2-D Poisson system.
 *
 * Implements the signature declared in cg.h.  Every vector operation (the
 * matrix-free stencil, the two dot products, and the three axpy/scale loops)
 * is parallelised with '#pragma omp parallel for'.  The outer CG iteration
 * loop is sequential because each iteration depends on the previous one.
 *
 * Determinism note: floating-point reductions with OpenMP can produce
 * different results depending on the number of threads because the order of
 * additions changes.  To guarantee bit-identical answers regardless of thread
 * count we accumulate each thread's partial sum into a private variable and
 * then sum those partial sums sequentially in thread-index order.
 */

#include "cg.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Deterministic parallel dot product.
 *
 * Each thread accumulates its own partial sum over a contiguous slice of the
 * array.  The partial sums are then added in thread-index order (sequential),
 * so the result is independent of the number of threads.
 * --------------------------------------------------------------------------- */
static double dot_det(const double * restrict u,
                      const double * restrict v,
                      long n)
{
    int nthreads;
    double *parts;

#pragma omp parallel shared(nthreads, parts)
    {
#pragma omp single
        {
            nthreads = omp_get_num_threads();
            parts = (double *)calloc((size_t)nthreads, sizeof(double));
        }

        int tid  = omp_get_thread_num();
        long lo  = (long)tid       * n / (long)nthreads;
        long hi  = (long)(tid + 1) * n / (long)nthreads;
        double s = 0.0;
        for (long k = lo; k < hi; k++) s += u[k] * v[k];
        parts[tid] = s;
    }

    double result = 0.0;
    for (int t = 0; t < nthreads; t++) result += parts[t];
    free(parts);
    return result;
}

/* ---------------------------------------------------------------------------
 * apply_A  –  matrix-free 5-point Poisson stencil
 *   (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1]
 * Neighbours outside the m×m grid contribute zero (Dirichlet BCs).
 * --------------------------------------------------------------------------- */
static void apply_A(double * restrict q,
                    const double * restrict v,
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

    /* r = b;  p = r;  rr = dot(r, r) */
    memcpy(r, b, sizeof(double) * (size_t)n);
    memcpy(p, b, sizeof(double) * (size_t)n);

    double rr = dot_det(r, r, n);

    /* CG iterations */
    for (int iter = 0; iter < iters; iter++) {

        /* q = A p */
        apply_A(q, p, m);

        /* alpha = rr / dot(p, q) */
        double pq    = dot_det(p, q, n);
        double alpha = rr / pq;

        /* x += alpha * p
         * r -= alpha * q  */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }

        /* rr_new = dot(r, r) */
        double rr_new = dot_det(r, r, n);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
#pragma omp parallel for schedule(static)
        for (long k = 0; k < n; k++) p[k] = r[k] + beta * p[k];
    }

    free(r);
    free(p);
    free(q);
}
