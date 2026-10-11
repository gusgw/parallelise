#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "cg.h"

static double local_dot(const double *a, const double *b, long n) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (m * rank) / size;
    long hi = (m * (rank + 1)) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;

    double *x_loc = (double *)calloc(local_n, sizeof(double));
    double *r_loc = (double *)malloc(local_n * sizeof(double));
    double *p_loc = (double *)malloc(local_n * sizeof(double));
    double *q_loc = (double *)malloc(local_n * sizeof(double));

    const double *b_start = b + lo * m;
    memcpy(r_loc, b_start, local_n * sizeof(double));
    memcpy(p_loc, r_loc, local_n * sizeof(double));
    memset(x_loc, 0, local_n * sizeof(double));

    double *recv_up = (double *)malloc(m * sizeof(double));
    double *recv_down = (double *)malloc(m * sizeof(double));

    double rr_local = local_dot(r_loc, r_loc, local_n);
    double rr_global;
    MPI_Allreduce(&rr_local, &rr_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int it = 0; it < iters; ++it) {
        int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        /* Simple halo exchange using tag 0 */
        MPI_Sendrecv(p_loc, m, MPI_DOUBLE, up, 0,
                     recv_up, m, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(p_loc + (local_rows - 1) * m, m, MPI_DOUBLE, down, 0,
                     recv_down, m, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (long i = 0; i < local_rows; ++i) {
            long global_i = lo + i;
            for (long j = 0; j < m; ++j) {
                long idx = i * m + j;
                double s = 4.0 * p_loc[idx];
                if (global_i > 0) {
                    if (i > 0) s -= p_loc[(i - 1) * m + j];
                    else s -= recv_up[j];
                }
                if (global_i < m - 1) {
                    if (i < local_rows - 1) s -= p_loc[(i + 1) * m + j];
                    else s -= recv_down[j];
                }
                if (j > 0) s -= p_loc[i * m + (j - 1)];
                if (j < m - 1) s -= p_loc[i * m + (j + 1)];
                q_loc[idx] = s;
            }
        }

        double dot_pq_local = local_dot(p_loc, q_loc, local_n);
        double dot_pq_global;
        MPI_Allreduce(&dot_pq_local, &dot_pq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr_global / dot_pq_global;
        for (long i = 0; i < local_n; ++i) {
            x_loc[i] += alpha * p_loc[i];
            r_loc[i] -= alpha * q_loc[i];
        }
        double rr_new_local = local_dot(r_loc, r_loc, local_n);
        double rr_new_global;
        MPI_Allreduce(&rr_new_local, &rr_new_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double beta = rr_new_global / rr_global;
        rr_global = rr_new_global;
        for (long i = 0; i < local_n; ++i) {
            p_loc[i] = r_loc[i] + beta * p_loc[i];
        }
    }

    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int local_n_int = (int)local_n;
    MPI_Gather(&local_n_int, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Allgatherv(x_loc, local_n_int, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(x_loc);
    free(r_loc);
    free(p_loc);
    free(q_loc);
    free(recv_up);
    free(recv_down);
    free(recvcounts);
    free(displs);
}
