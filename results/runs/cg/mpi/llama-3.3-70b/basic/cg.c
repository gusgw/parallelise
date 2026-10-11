#include "cg.h"
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function to compute the matrix-vector product
void apply_A(double *v, double *q, int m) {
    int i, j, idx;
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

// Function to compute the dot product
double dot(double *u, double *v, int n) {
    double s = 0.0;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize r and p
    for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr = 0.0;
    for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
        rr += r[i] * r[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        apply_A(p, q, m);
        double alpha = 0.0;
        for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
            alpha += r[i] * r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = alpha / dot(p, q, n);
        for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
            x[i] += alpha * p[i];
        }
        for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
            r[i] -= alpha * q[i];
        }
        double rr_new = 0.0;
        for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
            rr_new += r[i] * r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double beta = rr_new / rr;
        rr = rr_new;
        for (int i = rank * n / size; i < (rank + 1) * n / size; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
