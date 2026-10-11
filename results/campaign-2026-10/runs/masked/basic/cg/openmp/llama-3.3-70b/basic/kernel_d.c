#include "kernel_d.h"
#include "reference.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    double *p = (double *)malloc(m * sizeof(double));
    double *r = (double *)malloc(m * sizeof(double));
    double *Ap = (double *)malloc(m * sizeof(double));
    double rsold = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double temp = 0.0;

    #pragma omp parallel for
    for (long i = 0; i < m; i++) {
        r[i] = b[i];
        p[i] = 0.0;
        Ap[i] = 0.0;
    }

    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for reduction(+:rsold)
        for (long i = 0; i < m; i++) {
            rsold += r[i] * r[i];
        }

        if (rsold < REF_TOL * REF_TOL) {
            break;
        }

        #pragma omp parallel for
        for (long i = 0; i < m; i++) {
            Ap[i] = 0.0;
        }

        #pragma omp parallel for
        for (long i = 0; i < m; i++) {
            for (long j = 0; j < m; j++) {
                Ap[i] += (i == j) ? 4.0 : (i == j - 1 || i == j + 1) ? -1.0 : 0.0;
            }
        }

        #pragma omp parallel for reduction(+:temp)
        for (long i = 0; i < m; i++) {
            temp += p[i] * Ap[i];
        }

        alpha = rsold / temp;

        #pragma omp parallel for
        for (long i = 0; i < m; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        beta = 0.0;

        #pragma omp parallel for reduction(+:beta)
        for (long i = 0; i < m; i++) {
            beta += r[i] * r[i];
        }

        beta /= rsold;

        #pragma omp parallel for
        for (long i = 0; i < m; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(p);
    free(r);
    free(Ap);
}
