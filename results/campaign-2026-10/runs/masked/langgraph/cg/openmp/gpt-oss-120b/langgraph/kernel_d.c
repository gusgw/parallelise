#include <stdlib.h>
#include <math.h>
#include "kernel_d.h"

/* Apply the 5‑point stencil matrix A to vector v (size m*m).
   The matrix corresponds to the discrete Laplacian with Dirichlet
   boundaries: for each interior point we compute
       (4*v[i] - v[i-1] - v[i+1] - v[i-m] - v[i+m])
   taking care of missing neighbours at the domain edges.
   The result is stored in the pre‑allocated array q.
*/
static void apply_A(const double *v, double *q, long m)
{
    long n = m * m;
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < n; ++idx) {
        long i = idx / m;   // row
        long j = idx % m;   // column
        double s = 4.0 * v[idx];
        if (i > 0)     s -= v[idx - m];
        if (i < m-1)   s -= v[idx + m];
        if (j > 0)     s -= v[idx - 1];
        if (j < m-1)   s -= v[idx + 1];
        q[idx] = s;
    }
}

static double dot(const double *u, const double *v, long n)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < n; ++i) {
        sum += u[i] * v[i];
    }
    return sum;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    // allocate temporary vectors
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));
    if (!r || !p || !q) {
        // allocation failure – abort (the test harness will treat a crash as failure)
        abort();
    }

    // initialise x to zero and copy b into r and p
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    double rr = dot(r, r, n);

    for (int it = 0; it < iters; ++it) {
        apply_A(p, q, m);
        double pq = dot(p, q, n);
        double alpha = rr / pq;

        // x = x + alpha * p ; r = r - alpha * q
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;

        // p = r + beta * p
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
