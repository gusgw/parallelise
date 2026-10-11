#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "cg.h"

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long rows_per_rank = m / size;
    long rem = m % size;
    long my_rows = rows_per_rank + (rank < rem ? 1 : 0);
    long my_lo = rank * rows_per_rank + (rank < rem ? rank : rem);

    long my_n = my_rows * m;
    double *r_local = (double *)malloc(my_n * sizeof(double));
    double *p_local = (double *)malloc(my_n * sizeof(double));
    double *q_local = (double *)malloc(my_n * sizeof(double));
    double *x_local = (double *)malloc(my_n * sizeof(double));

    // Initial x = 0, r = b, p = r
    memset(x_local, 0, my_n * sizeof(double));
    for (long i = 0; i < my_n; i++) {
        r_local[i] = b[my_lo * m + i];
        p_local[i] = r_local[i];
    }

    double rr_local = 0.0;
    for (long i = 0; i < my_n; i++) rr_local += r_local[i] * r_local[i];
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double *p_top = (double *)malloc(m * sizeof(double));
    double *p_bot = (double *)malloc(m * sizeof(double));

    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    for (int iter = 0; iter < iters; iter++) {
        // Halo exchange for p
        // Send top row to up, receive bottom row from down
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, up, 0, 
                     p_bot, m, MPI_DOUBLE, down, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send bottom row to down, receive top row from up
        MPI_Sendrecv(p_local + (my_rows - 1) * m, m, MPI_DOUBLE, down, 1, 
                     p_top, m, MPI_DOUBLE, up, 1, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // q = A p
        for (long i = 0; i < my_rows; i++) {
            long global_i = my_lo + i;
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                double val = 4.0 * p_local[idx];
                if (global_i > 0) {
                    val -= (i == 0) ? p_top[j] : p_local[idx - m];
                }
                if (global_i < m - 1) {
                    val -= (i == my_rows - 1) ? p_bot[j] : p_local[idx + m];
                }
                if (j > 0) val -= p_local[idx - 1];
                if (j < m - 1) val -= p_local[idx + 1];
                q_local[idx] = val;
            }
        }

        double pq_local = 0.0;
        for (long i = 0; i < my_n; i++) pq_local += p_local[i] * q_local[i];
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rr / pq;
        for (long i = 0; i < my_n; i++) {
            x_local[i] += alpha * p_local[i];
            r_local[i] -= alpha * q_local[i];
        }

        double rr_new_local = 0.0;
        for (long i = 0; i < my_n; i++) rr_new_local += r_local[i] * r_local[i];
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;
        for (long i = 0; i < my_n; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }

    // Assemble full x
    int *counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int total_n = 0;
    for (int r = 0; r < size; r++) {
        int r_rows = (int)(m / size + (r < m % size ? 1 : 0));
        counts[r] = r_rows * (int)m;
        displs[r] = total_n;
        total_n += counts[r];
    }

    MPI_Allgatherv(x_local, (int)my_n, MPI_DOUBLE, x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(r_local); free(p_local); free(q_local); free(x_local);
    free(p_top); free(p_bot); free(counts); free(displs);
}
