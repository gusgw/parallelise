#include "kernel_d.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* Apply 5-point stencil: 4*v[i,j] - neighbors */
static void apply_A(const double *v, double *q, long m)
{
    long n = m * m;
    #pragma omp parallel for
    for (long idx = 0; idx < n; idx++) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * v[idx];
        if (i > 0)
            s -= v[idx - m];
        if (i < m - 1)
            s -= v[idx + m];
        if (j > 0)
            s -= v[idx - 1];
        if (j < m - 1)
            s -= v[idx + 1];
        q[idx] = s;
    }
}

/* Dot product with reduction */
static double dot(const double *u, const double *v, long n)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (long k = 0; k < n; k++) {
        s += u[k] * v[k];
    }
    return s;
}

/* x = x + alpha * p */
static void axpy(double *x, const double *p, double alpha, long n)
{
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] += alpha * p[k];
    }
}

/* r = r - alpha * q */
static void axmy(double *r, const double *q, double alpha, long n)
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
    
    /* Initialize x = 0, r = b, p = r */
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
    }
    
    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);
    
    if (!r || !p || !q) {
        free(r);
        free(p);
        free(q);
        return;
    }
    
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        r[k] = b[k];
        p[k] = b[k];
    }
    
    double rr = dot(r, r, n);
    
    for (int iter = 0; iter < iters; iter++) {
        apply_A(p, q, m);
        double pq = dot(p, q, n);
        double alpha = rr / pq;
        
        axpy(x, p, alpha, n);
        axmy(r, q, alpha, n);
        
        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        
        update_p(p, r, beta, n);
    }
    
    free(r);
    free(p);
    free(q);
}
