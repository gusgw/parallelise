#include "cg.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Matrix-free 5-point Poisson operator: neighbours off the edge are zero (Dirichlet).
 * (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1].
 * Parallelised with OpenMP over the grid. */
static void apply_A(const double *v, double *q, long m)
{
    long n = m * m;
    #pragma omp parallel for
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

/* Dot product of two vectors of length n, parallelised with OpenMP reduction. */
static double dot(const double *u, const double *v, long n)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (long k = 0; k < n; k++) {
        s += u[k] * v[k];
    }
    return s;
}

/* x += alpha * p (AXPY), parallelised with OpenMP. */
static void axpy_x(double *x, const double *p, double alpha, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] += alpha * p[k];
    }
}

/* r -= alpha * q (AXPY), parallelised with OpenMP. */
static void axpy_r(double *r, const double *q, double alpha, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        r[k] -= alpha * q[k];
    }
}

/* p = r + beta * p, parallelised with OpenMP. */
static void update_p(double *p, const double *r, double beta, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        p[k] = r[k] + beta * p[k];
    }
}

/* Conjugate gradient for the 2D Poisson system on an m x m grid.
 * Standard CG from x = 0, run for a FIXED number of iterations.
 * Parallelised with OpenMP inside each vector operation. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    
    /* Initialize x = 0, r = b, p = r */
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
    }
    
    /* Allocate workspace for r, p, q */
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    
    /* Initialize r = b, p = r */
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        r[k] = b[k];
        p[k] = r[k];
    }
    
    double rr = dot(r, r, n);
    
    for (int iter = 0; iter < iters; iter++) {
        apply_A(p, q, m);
        double pq = dot(p, q, n);
        double alpha = rr / pq;
        
        axpy_x(x, p, alpha, n);
        axpy_r(r, q, alpha, n);
        
        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        
        update_p(p, r, beta, n);
    }
    
    free(r);
    free(p);
    free(q);
}
