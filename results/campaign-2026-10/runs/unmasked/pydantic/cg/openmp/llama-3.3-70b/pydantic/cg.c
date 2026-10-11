#include "cg.h"
#include <omp.h>
#include <stdlib.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    double *r = (double *)malloc(m * m * sizeof(double));
    double *p = (double *)malloc(m * m * sizeof(double));
    double *q = (double *)malloc(m * m * sizeof(double));
    double rr = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    long i;
    
    #pragma omp parallel for
    for (i = 0; i < m * m; i++) {
        r[i] = b[i];
        p[i] = r[i];
        x[i] = 0.0;
    }
    
    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for reduction(+:rr)
        for (i = 0; i < m * m; i++) {
            rr += r[i] * r[i];
        }
        
        #pragma omp parallel for
        for (i = 0; i < m * m; i++) {
            long idx = i / m;
            long idy = i % m;
            q[i] = 4 * p[i];
            if (idx > 0) q[i] -= p[i - m];
            if (idx < m - 1) q[i] -= p[i + m];
            if (idy > 0) q[i] -= p[i - 1];
            if (idy < m - 1) q[i] -= p[i + 1];
        }
        
        double pq = 0.0;
        #pragma omp parallel for reduction(+:pq)
        for (i = 0; i < m * m; i++) {
            pq += p[i] * q[i];
        }
        
        alpha = rr / pq;
        
        #pragma omp parallel for
        for (i = 0; i < m * m; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        
        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (i = 0; i < m * m; i++) {
            rr_new += r[i] * r[i];
        }
        
        beta = rr_new / rr;
        rr = rr_new;
        
        #pragma omp parallel for
        for (i = 0; i < m * m; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }
    
    free(r);
    free(p);
    free(q);
}
