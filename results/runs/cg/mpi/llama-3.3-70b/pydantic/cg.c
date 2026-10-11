#include "cg.h"
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    int n = m * m;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));

    // Initialize x, r, p
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr = 0.0;
    for (int i = 0; i < n; i++) {
        rr += r[i] * r[i];
    }

    for (int iter = 0; iter < iters; iter++) {
        // Apply A to p
        for (int i = 0; i < n; i++) {
            q[i] = 4.0 * p[i];
            if (i >= m) {
                q[i] -= p[i - m];
            }
            if (i < n - m) {
                q[i] -= p[i + m];
            }
            if (i % m > 0) {
                q[i] -= p[i - 1];
            }
            if (i % m < m - 1) {
                q[i] -= p[i + 1];
            }
        }

        double alpha = 0.0;
        for (int i = 0; i < n; i++) {
            alpha += p[i] * q[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        alpha = rr / alpha;

        // Update x and r
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = 0.0;
        for (int i = 0; i < n; i++) {
            rr_new += r[i] * r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        // Update p
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
