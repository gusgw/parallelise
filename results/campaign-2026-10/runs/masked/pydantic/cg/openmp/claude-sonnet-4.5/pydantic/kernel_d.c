#include "kernel_d.h"
#include <stdlib.h>
#include <string.h>

static void apply_A(double *q, const double *v, long m)
{
    #pragma omp parallel for
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
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
}

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
    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);
    
    #pragma omp parallel for
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
    }
    
    double rr = dot(r, r, n);
    
    for (int iter = 0; iter < iters; iter++) {
        apply_A(q, p, m);
        double alpha = rr / dot(p, q, n);
        
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
        }
        
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            r[k] -= alpha * q[k];
        }
        
        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        
        #pragma omp parallel for
        for (long k = 0; k < n; k++) {
            p[k] = r[k] + beta * p[k];
        }
    }
    
    free(r);
    free(p);
    free(q);
}
