/* The serial routine of sparse_cg.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 8, 32, 128, 320]. */
#include "sparse_cg.h"
#include <math.h>
#include <stdlib.h>

static void multiply(const int *indptr, const int *indices, const double *data,
                     const double *v, double *result, long n)
{
    /* Compute A*v where A is in CSR format.
     * For each row i, result[i] = sum of data[j] * v[indices[j]] for j in [indptr[i], indptr[i+1])
     */
    for (long i = 0; i < n; i++) {
        double sum = 0.0;
        for (long j = indptr[i]; j < indptr[i + 1]; j++) {
            sum += data[j] * v[indices[j]];
        }
        result[i] = sum;
    }
}

static double dot_product(const double *a, const double *b, long n)
{
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void sparse_cg(const int *indptr, const int *indices, const double *data, const double *b,
               double rtol, long max_iter, double *x, long np1, long nnz, long n)
{
    (void)np1;  /* Not needed, we use n instead */
    (void)nnz;  /* Not needed directly */

    /* Allocate workspace */
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));

    if (!r || !p || !q) {
        /* Handle allocation failure - return zeros */
        for (long i = 0; i < n; i++) {
            x[i] = 0.0;
        }
        free(r);
        free(p);
        free(q);
        return;
    }

    /* Initialize x = 0, r = b, p = r */
    for (long i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    double rr = dot_product(r, r, n);
    double b_norm_sq = dot_product(b, b, n);
    double limit = rtol * rtol * b_norm_sq;

    for (long iter = 0; iter < max_iter; iter++) {
        if (rr <= limit) {
            break;
        }

        /* q = A * p */
        multiply(indptr, indices, data, p, q, n);

        /* alpha = rr / (p @ q) */
        double pdotq = dot_product(p, q, n);
        if (pdotq == 0.0) {
            break;  /* Avoid division by zero */
        }
        double alpha = rr / pdotq;

        /* x += alpha * p */
        /* r -= alpha * q */
        for (long i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        /* rr_new = r @ r */
        double rr_new = dot_product(r, r, n);

        /* p = r + (rr_new / rr) * p */
        double beta = rr_new / rr;
        for (long i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }

    free(r);
    free(p);
    free(q);
}
