#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kernel_d.h"

/* Helper: dot product of two local vectors */
static double local_dot(const double *a, const double *b, long n) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine row range for this rank */
    long lo_row = (m * rank) / size;
    long hi_row = (m * (rank + 1)) / size;
    long local_rows = hi_row - lo_row;
    long local_n = local_rows * m;

    /* Allocate local vectors */
    double *x_local = (double *)calloc(local_n, sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));
    double *top_halo   = (double *)calloc(m, sizeof(double));
    double *bottom_halo = (double *)calloc(m, sizeof(double));

    /* Initialize local portions of r and p from global b */
    for (long i = 0; i < local_rows; ++i) {
        long global_row = lo_row + i;
        memcpy(r_local + i * m, b + global_row * m, m * sizeof(double));
    }
    memcpy(p_local, r_local, local_n * sizeof(double));
    memset(x_local, 0, local_n * sizeof(double));

    /* Initial residual norm (global) */
    double rr_local = local_dot(r_local, r_local, local_n);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int it = 0; it < iters; ++it) {
        if (local_rows > 0) {
            int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
            int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

            /* Exchange halo rows of p */
            MPI_Sendrecv(p_local,                m, MPI_DOUBLE, up,   0,
                         bottom_halo,           m, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(p_local + (local_rows - 1) * m, m, MPI_DOUBLE, down, 1,
                         top_halo,               m, MPI_DOUBLE, up,   1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Apply A to p_local -> q_local */
            for (long i = 0; i < local_rows; ++i) {
                for (long j = 0; j < m; ++j) {
                    long idx = i * m + j;
                    double s = 4.0 * p_local[idx];
                    /* up neighbor */
                    if (i > 0) {
                        s -= p_local[idx - m];
                    } else {
                        s -= top_halo[j];
                    }
                    /* down neighbor */
                    if (i < local_rows - 1) {
                        s -= p_local[idx + m];
                    } else {
                        s -= bottom_halo[j];
                    }
                    /* left neighbor */
                    if (j > 0) s -= p_local[idx - 1];
                    /* right neighbor */
                    if (j < m - 1) s -= p_local[idx + 1];
                    q_local[idx] = s;
                }
            }
        }

        /* Compute dot(p,q) globally */
        double pq_local = local_dot(p_local, q_local, local_n);
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* Update x and r locally */
        for (long i = 0; i < local_n; ++i) {
            x_local[i] += alpha * p_local[i];
            r_local[i] -= alpha * q_local[i];
        }

        /* New residual norm */
        double rr_new_local = local_dot(r_local, r_local, local_n);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double beta = rr_new / rr;
        rr = rr_new;

        /* Update p */
        for (long i = 0; i < local_n; ++i) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }

    /* Gather the full solution onto all ranks */
    int local_count = (int)local_n;
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    MPI_Allgatherv(x_local, local_count, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    free(recvcounts);
    free(displs);

    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(top_halo);
    free(bottom_halo);
}
