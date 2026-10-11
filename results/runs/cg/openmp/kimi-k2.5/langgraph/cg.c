#include "cg.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Apply the 5-point Poisson stencil: (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1]
 * Neighbors off the edge contribute zero (Dirichlet boundaries). */
static void apply_A(double *q, const double *v, long m)
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

/* Compute dot product u^T v */
static double dot(const double *u, const double *v, long n)
{
    double s = 0.0;
    
    #pragma omp parallel for reduction(+:s)
    for (long k = 0; k < n; k++) {
        s += u[k] * v[k];
    }
    
    return s;
}

/* x += alpha * p */
static void axpy_x(double *x, const double *p, double alpha, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] += alpha * p[k];
    }
}

/* r -= alpha * q */
static void axpy_r(double *r, const double *q, double alpha, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        r[k] -= alpha * q[k];
    }
}

/* p = r + beta * p */
static void update_p(double *p, const double *r, double beta, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        p[k] = r[k] + beta * p[k];
    }
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    
    /* Allocate workspace */
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    
    if (!r || !p || !q) {
        free(r);
        free(p);
        free(q);
        return;
    }
    
    /* Initialize: x = 0, r = b, p = r */
    memset(x, 0, n * sizeof(double));
    memcpy(r, b, n * sizeof(double));
    memcpy(p, r, n * sizeof(double));
    
    double rr = dot(r, r, n);
    
    for (int iter = 0; iter < iters; iter++) {
        /* q = A * p */
        apply_A(q, p, m);
        
        /* alpha = rr / dot(p, q) */
        double pdotq = dot(p, q, n);
        double alpha = rr / pdotq;
        
        /* x += alpha * p */
        axpy_x(x, p, alpha, n);
        
        /* r -= alpha * q */
        axpy_r(r, q, alpha, n);
        
        /* rr_new = dot(r, r) */
        double rr_new = dot(r, r, n);
        
        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        
        /* p = r + beta * p */
        update_p(p, r, beta, n);
        
        rr = rr_new;
    }
    
    free(r);
    free(p);
    free(q);
}
