/* My own OpenMP routine for sparse_cg, written only to test the notebook; not a known-good routine. */
#include <stdlib.h>

#include "sparse_cg.h"

void sparse_cg(const int *indptr, const int *indices, const double *data, const double *b,
               double rtol, long max_iter, double *x, long np1, long nnz, long n)
{
    (void)np1;
    (void)nnz;
    double *r = malloc(sizeof(double) * (size_t)n);
    double *p = malloc(sizeof(double) * (size_t)n);
    double *q = malloc(sizeof(double) * (size_t)n);
    if (r == NULL || p == NULL || q == NULL) {
        free(r);
        free(p);
        free(q);
        return;
    }
    double rr = 0.0;
#pragma omp parallel for reduction(+ : rr)
    for (long i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
        rr += b[i] * b[i];
    }
    double limit = rtol * rtol * rr;
    for (long step = 0; step < max_iter; step++) {
        if (rr <= limit) {
            break;
        }
        double pq = 0.0;
#pragma omp parallel for reduction(+ : pq)
        for (long i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = indptr[i]; k < indptr[i + 1]; k++) {
                sum += data[k] * p[indices[k]];
            }
            q[i] = sum;
            pq += p[i] * sum;
        }
        double alpha = rr / pq;
        double rr_new = 0.0;
#pragma omp parallel for reduction(+ : rr_new)
        for (long i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            rr_new += r[i] * r[i];
        }
        double beta = rr_new / rr;
#pragma omp parallel for
        for (long i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    free(r);
    free(p);
    free(q);
}
