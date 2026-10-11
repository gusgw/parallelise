#include "kernel_d.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

/* Apply 2D Laplacian (5-point stencil) to vector v
 * A has 4 on diagonal, -1 for each neighbor (up, down, left, right)
 */
static void apply_A(double *q, const double *v, long m)
{
    long n = m * m;
    
    #pragma omp parallel for
    for (long idx = 0; idx < n; idx++) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * v[idx];
        if (i > 0) {
            s -= v[idx - m];
        }
        if (i < m - 1) {
            s -= v[idx + m];
        }
        if (j > 0) {
            s -= v[idx - 1];
        }
        if (j < m - 1) {
            s -= v[idx + 1];
        }
        q[idx] = s;
    }
}

/* Compute dot product of u and v */
static double dot(const double *u, const double *v, long n)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (long k = 0; k < n; k++) {
        s += u[k] * v[k];
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    
    /* Allocate temporary arrays */
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
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = r[k];
    }
    
    double rr = dot(r, r, n);
    
    for (int iter = 0; iter < iters; iter++) {
        /* q = A * p */
        apply_A(q, p, m);
        
        /* alpha = rr / dot(p, q) */
        double pdotq = dot(p, q, n);
        double alpha = rr / pdotq;
        
        /* x = x + alpha * p */
        /* r = r - alpha * q */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }
        
        /* rr_new = dot(r, r) */
        double rr_new = dot(r, r, n);
        
        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        
        /* p = r + beta * p */
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            p[k] = r[k] + beta * p[k];
        }
        
        rr = rr_new;
    }
    
    free(r);
    free(p);
    free(q);
}
