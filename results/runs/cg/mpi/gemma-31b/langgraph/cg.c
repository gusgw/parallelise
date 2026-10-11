#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "cg.h"

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (m / size) * rank;
    long hi = (m / size) * (rank + 1);
    if (rank == size - 1) hi = m;

    long local_rows = hi - lo;
    long local_n = local_rows * m;

    double *local_x = (double *)calloc(local_n, sizeof(double));
    double *local_r = (double *)malloc(local_n * sizeof(double));
    double *local_p = (double *)malloc(local_n * sizeof(double));
    double *local_q = (double *)malloc(local_n * sizeof(double));

    for (long i = 0; i < local_n; i++) {
        local_r[i] = b[lo * m + i];
        local_p[i] = local_r[i];
    }

    double local_rr = 0.0;
    for (long i = 0; i < local_n; i++) local_rr += local_r[i] * local_r[i];
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double *p_up = (double *)malloc(m * sizeof(double));
    double *p_down = (double *)malloc(m * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        // Send top row to up, receive bottom row from down into p_down
        MPI_Sendrecv(local_p, m, MPI_DOUBLE, up, 0,
                     p_down, m, MPI_DOUBLE, down, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send bottom row to down, receive top row from up into p_up
        MPI_Sendrecv(local_p + (local_rows - 1) * m, m, MPI_DOUBLE, down, 0,
                     p_up, m, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (long i = 0; i < local_rows; i++) {
            long global_i = lo + i;
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                double val = 4.0 * local_p[idx];
                if (global_i > 0) {
                    val -= (i == 0) ? p_down[j] : local_p[idx - m];
                }
                if (global_i < m - 1) {
                    val -= (i == local_rows - 1) ? p_up[j] : local_p[idx + m];
                }
                if (j > 0) val -= local_p[idx - 1];
                if (j < m - 1) val -= local_p[idx + 1];
                local_q[idx] = val;
            }
        }

        double local_pq = 0.0;
        for (long i = 0; i < local_n; i++) local_pq += local_p[i] * local_q[i];
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rr / pq;
        for (long i = 0; i < local_n; i++) {
            local_x[i] += alpha * local_p[i];
            local_r[i] -= alpha * local_q[i];
        }

        double local_rr_new = 0.0;
        for (long i = 0; i < local_n; i++) local_rr_new += local_r[i] * local_r[i];
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;
        for (long i = 0; i < local_n; i++) {
            local_p[i] = local_r[i] + beta * local_p[i];
        }
    }

    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        long l = (m / size) * i;
        long h = (m / size) * (i + 1);
        if (i == size - 1) h = m;
        recv_counts[i] = (int)((h - l) * m);
        displs[i] = (int)(l * m);
    }

    MPI_Allgatherv(local_x, (int)local_n, MPI_DOUBLE, x, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_x); free(local_r); free(local_p); free(local_q);
    free(p_up); free(p_down); free(recv_counts); free(displs);
}
