#include "cg.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void cg_solve(double *x, double *b, int m, int iters) {
    int n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    double rr, alpha, beta;
    int i, j, idx;
    
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }
    
    rr = 0.0;
    #pragma omp parallel for reduction(+:rr)
    for (i = 0; i < n; i++) {
        rr += r[i] * r[i];
    }
    
    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            q[i] = 0.0;
        }
        
        #pragma omp parallel for
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                idx = i * m + j;
                double s = 4.0 * p[idx];
                if (i > 0)     s -= p[idx - m];
                if (i < m - 1) s -= p[idx + m];
                if (j > 0)     s -= p[idx - 1];
                if (j < m - 1) s -= p[idx + 1];
                q[idx] = s;
            }
        }
        
        alpha = 0.0;
        #pragma omp parallel for reduction(+:alpha)
        for (i = 0; i < n; i++) {
            alpha += p[i] * q[i];
        }
        alpha = rr / alpha;
        
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            r[i] -= alpha * q[i];
        }
        
        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (i = 0; i < n; i++) {
            rr_new += r[i] * r[i];
        }
        beta = rr_new / rr;
        rr = rr_new;
        
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }
    
    free(r);
    free(p);
    free(q);
}
