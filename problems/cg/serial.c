/* The serial routine of cg.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 8, 64, 256, 1024]. */
#include "cgsolve.h"
#include <stdlib.h>
#include <string.h>

/* Apply the five-point operator to a grid, counting values beyond the edge as zero. */
static void poisson_operator(double *out, const double *v, long m)
{
    long i, j;
    long idx;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            idx = i * m + j;
            out[idx] = 4.0 * v[idx];
            
            /* Subtract neighbor above (i-1, j) if it exists */
            if (i > 0) {
                out[idx] -= v[(i - 1) * m + j];
            }
            
            /* Subtract neighbor below (i+1, j) if it exists */
            if (i < m - 1) {
                out[idx] -= v[(i + 1) * m + j];
            }
            
            /* Subtract neighbor left (i, j-1) if it exists */
            if (j > 0) {
                out[idx] -= v[i * m + (j - 1)];
            }
            
            /* Subtract neighbor right (i, j+1) if it exists */
            if (j < m - 1) {
                out[idx] -= v[i * m + (j + 1)];
            }
        }
    }
}

/* Compute dot product of two vectors of length m*m */
static double vdot(const double *a, const double *b, long m)
{
    long i, n = m * m;
    double sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    long i;
    int iter;
    double *r, *p, *q;
    double rr, rr_new, alpha, beta;
    
    /* Allocate temporary arrays */
    r = (double *)malloc(n * sizeof(double));
    p = (double *)malloc(n * sizeof(double));
    q = (double *)malloc(n * sizeof(double));
    
    /* Initialize x = 0 */
    memset(x, 0, n * sizeof(double));
    
    /* r = b (since x = 0, residual is b - A*0 = b) */
    memcpy(r, b, n * sizeof(double));
    
    /* p = r */
    memcpy(p, r, n * sizeof(double));
    
    /* rr = r . r */
    rr = vdot(r, r, m);
    
    for (iter = 0; iter < iters; iter++) {
        /* q = A * p */
        poisson_operator(q, p, m);
        
        /* alpha = rr / (p . q) */
        alpha = rr / vdot(p, q, m);
        
        /* x += alpha * p */
        /* r -= alpha * q */
        for (i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        
        /* rr_new = r . r */
        rr_new = vdot(r, r, m);
        
        /* beta = rr_new / rr */
        beta = rr_new / rr;
        
        /* p = r + beta * p */
        for (i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        /* rr = rr_new */
        rr = rr_new;
    }
    
    /* Free temporary arrays */
    free(r);
    free(p);
    free(q);
}
