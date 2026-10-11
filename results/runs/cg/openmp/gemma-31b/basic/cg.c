#include "cg.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    long n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));

    if (!r || !p || !q) {
        free(r); free(p); free(q);
        return;
    }

    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr = 0.0;
    #pragma omp parallel for reduction(+:rr)
    for (long i = 0; i < n; i++) {
        rr += r[i] * r[i];
    }

    for (int iter = 0; iter < iters; iter++) {
        // q = A p
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

        double pq = 0.0;
        #pragma omp parallel for reduction(+:pq)
        for (long i = 0; i < n; i++) {
            pq += p[i] * q[i];
        }

        double alpha = rr / pq;

        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (long i = 0; i < n; i++) {
            rr_new += r[i] * r[i];
        }

        double beta = rr_new / rr;
        rr = rr_new;

        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
