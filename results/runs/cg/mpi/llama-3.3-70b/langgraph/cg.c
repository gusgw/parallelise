#include "cg.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

// Function to compute the matrix-free 5-point Poisson operator
void apply_A(double *v, int m, double *q) {
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

// Function to compute the dot product of two vectors
double dot(double *u, double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += u[i] * v[i];
    }
    return s;
}

// Function to solve the system using conjugate gradient
void cgsolve(double *x, const double *b, int m, int iters) {
    int n = m * m;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = n / size;
    int start = rank * block_size;
    int end = (rank + 1) * block_size;

    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(n * sizeof(double));

    // Initialize r and p
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr = dot(r, r, n);
    for (int iter = 0; iter < iters; iter++) {
        apply_A(p, m, q);
        double alpha = rr / dot(p, q, n);
        for (int i = start; i < end; i++) {
            x[i] += alpha * p[i];
        }
        for (int i = start; i < end; i++) {
            r[i] -= alpha * q[i];
        }
        double rr_new = dot(r, r, n);
        double beta = rr_new / rr;
        rr = rr_new;
        for (int i = start; i < end; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(q);
}
