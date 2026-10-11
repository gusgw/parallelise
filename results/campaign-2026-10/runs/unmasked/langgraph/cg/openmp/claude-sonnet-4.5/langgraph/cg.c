#include "cg.h"
#include <stdlib.h>
#include <string.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations (no convergence
 * test -- the run must be deterministic). The operator is the 5-point stencil with
 * Dirichlet boundaries OUTSIDE the grid, so a neighbour off the edge contributes zero:
 *     (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1].
 * Standard CG from x = 0:  r = b; p = r; rr = dot(r,r);  then `iters` times:
 *     q = A p;  alpha = rr / dot(p,q);  x += alpha p;  r -= alpha q;
 *     rr_new = dot(r,r);  beta = rr_new / rr;  rr = rr_new;  p = r + beta p.
 * On return x holds the iterate after `iters` iterations.
 *
 * Parallelised with OpenMP INSIDE each vector operation. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    
    /* Allocate working arrays */
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    
    /* Initialize x = 0, r = b, p = r */
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
    }
    
    /* rr = dot(r, r) */
    double rr = 0.0;
    #pragma omp parallel for reduction(+:rr)
    for (long k = 0; k < n; k++) {
        rr += r[k] * r[k];
    }
    
    /* CG iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* q = A p: 5-point stencil */
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
        
        /* alpha = rr / dot(p, q) */
        double pq = 0.0;
        #pragma omp parallel for reduction(+:pq)
        for (long k = 0; k < n; k++) {
            pq += p[k] * q[k];
        }
        double alpha = rr / pq;
        
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
        
        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;
        
        /* p = r + beta * p */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            p[k] = r[k] + beta * p[k];
        }
    }
    
    /* Clean up */
    free(r);
    free(p);
    free(q);
}
