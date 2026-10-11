#include "cg.h"
#include "reference.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;

    double *r = (double *) malloc(m * sizeof(double));
    double *p = (double *) malloc(m * sizeof(double));
    double *q = (double *) malloc(m * sizeof(double));

    // Initialize r and p
    for (long i = lo; i < hi; i++) {
        r[i] = b[i];
        p[i] = r[i];
    }

    double rr = 0.0;
    for (long i = lo; i < hi; i++) {
        rr += r[i] * r[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        // Exchange boundary rows of p
        double *buf = (double *) malloc(m * sizeof(double));
        if (rank > 0) {
            MPI_Sendrecv(p + lo - 1, 1, MPI_DOUBLE, rank - 1, 0, buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(p + hi, 1, MPI_DOUBLE, rank + 1, 0, buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute q = A p
        for (long i = lo; i < hi; i++) {
            q[i] = 4 * p[i] - (i > 0 ? p[i - 1] : 0) - (i < m - 1 ? p[i + 1] : 0);
        }

        double pq = 0.0;
        for (long i = lo; i < hi; i++) {
            pq += p[i] * q[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rr / pq;

        // Update x and r
        for (long i = lo; i < hi; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        double rr_new = 0.0;
        for (long i = lo; i < hi; i++) {
            rr_new += r[i] * r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        // Update p
        for (long i = lo; i < hi; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    // Gather x
    int *counts = (int *) malloc(size * sizeof(int));
    int *displs = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        counts[i] = m / size;
        displs[i] = i * counts[i];
    }
    if (rank == size - 1) {
        counts[size - 1] += m % size;
    }
    double *x_buf = (double *) malloc(m * sizeof(double));
    MPI_Allgatherv(x + lo, hi - lo, MPI_DOUBLE, x_buf, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    memcpy(x, x_buf, m * sizeof(double));

    free(r);
    free(p);
    free(q);
    free(x_buf);
    free(counts);
    free(displs);
}
