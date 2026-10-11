#include "cg.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Function to apply the 5-point Poisson operator
void apply_A(double *v, double *q, int m) {
    int i, j, idx;
    #pragma omp parallel for
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            idx = i * m + j;
            double s = 4.0 * v[idx];
            if (i > 0)     s -= v[idx - m];
            if (i < m - 1) s -= v[idx + m];
            if (j > 0)     s -= v[idx - 1];
            if (j < m - 1) s -= v[idx + 1];
            q[idx] = s;
        }
    }
}

// Function to compute the dot product of two vectors
double dot(double *u, double *v, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += u[i] * v[i];
    }
    return s;
}

// Conjugate gradient solver
void cgsolve(double *b, double *x, int m, int iters) {
    int n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    
    // Initialize r and p
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }
    
    double rr = dot(r, r, n);
    for (int iter = 0; iter < iters; iter++) {
        apply_A(p, q, m);
        double alpha = rr / dot(p, q, n);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            r[i] -= alpha * q[i];
        }
        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }
    
    free(r);
    free(p);
    free(q);
}
