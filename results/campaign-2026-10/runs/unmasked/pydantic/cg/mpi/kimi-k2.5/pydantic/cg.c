#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose rows across ranks: each rank owns rows [lo, hi) */
    long lo = (m * rank) / size;
    long hi = (m * (rank + 1)) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;

    /* Allocate local arrays */
    double *local_x = calloc(local_n, sizeof(double));
    double *local_r = malloc(local_n * sizeof(double));
    double *local_p = malloc(local_n * sizeof(double));
    double *local_q = malloc(local_n * sizeof(double));

    /* Halo buffer for p: one row above and one row below */
    double *p_above = malloc(m * sizeof(double));
    double *p_below = malloc(m * sizeof(double));

    /* Initialize: x = 0, r = b, p = r */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            long local_idx = (i - lo) * m + j;
            local_r[local_idx] = b[idx];
            local_p[local_idx] = local_r[local_idx];
        }
    }

    /* Compute initial rr = dot(r, r) */
    double local_rr = 0.0;
    for (long k = 0; k < local_n; k++) {
        local_rr += local_r[k] * local_r[k];
    }
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange boundary rows of p with neighbors */
        int rank_above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        /* Send row lo to rank_above, receive from rank_below into p_below */
        MPI_Sendrecv(local_p, m, MPI_DOUBLE, rank_above, 0,
                     p_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send row hi-1 to rank_below, receive from rank_above into p_above */
        MPI_Sendrecv(local_p + (local_rows - 1) * m, m, MPI_DOUBLE, rank_below, 1,
                     p_above, m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p: q = A * p */
        for (long i = lo; i < hi; i++) {
            for (long j = 0; j < m; j++) {
                long local_i = i - lo;
                long local_idx = local_i * m + j;

                double p_center = local_p[local_idx];
                double s = 4.0 * p_center;

                /* Up neighbor */
                if (i > lo) {
                    s -= local_p[(local_i - 1) * m + j];
                } else if (i > 0) {
                    s -= p_above[j];
                }

                /* Down neighbor */
                if (i < hi - 1) {
                    s -= local_p[(local_i + 1) * m + j];
                } else if (i < m - 1) {
                    s -= p_below[j];
                }

                /* Left neighbor */
                if (j > 0) {
                    s -= local_p[local_idx - 1];
                }

                /* Right neighbor */
                if (j < m - 1) {
                    s -= local_p[local_idx + 1];
                }

                local_q[local_idx] = s;
            }
        }

        /* Compute dot(p, q) */
        double local_pq = 0.0;
        for (long k = 0; k < local_n; k++) {
            local_pq += local_p[k] * local_q[k];
        }
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* alpha = rr / dot(p, q) */
        double alpha = rr / pq;

        /* x += alpha * p */
        for (long k = 0; k < local_n; k++) {
            local_x[k] += alpha * local_p[k];
        }

        /* r -= alpha * q */
        for (long k = 0; k < local_n; k++) {
            local_r[k] -= alpha * local_q[k];
        }

        /* rr_new = dot(r, r) */
        double local_rr_new = 0.0;
        for (long k = 0; k < local_n; k++) {
            local_rr_new += local_r[k] * local_r[k];
        }
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;

        /* rr = rr_new */
        rr = rr_new;

        /* p = r + beta * p */
        for (long k = 0; k < local_n; k++) {
            local_p[k] = local_r[k] + beta * local_p[k];
        }
    }

    /* Gather full x on all ranks using MPI_Allgatherv */
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        long i_lo = (m * i) / size;
        long i_hi = (m * (i + 1)) / size;
        long i_rows = i_hi - i_lo;
        recvcounts[i] = (int)(i_rows * m);
    }

    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(local_x, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Cleanup */
    free(local_x);
    free(local_r);
    free(local_p);
    free(local_q);
    free(p_above);
    free(p_below);
    free(recvcounts);
    free(displs);
}
