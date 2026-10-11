#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "cg.h"

void cgsolve(double *x, const double *b, long m, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = m * m;
    long lo = (n / m) * rank / size; // This is wrong, should be m * rank / size
    // Correcting: decompose rows across ranks
    long rows_per_rank = m / size;
    long rem = m % size;
    long my_rows = rows_per_rank + (rank < rem ? 1 : 0);
    long my_lo = rank * rows_per_rank + (rank < rem ? rank : rem);
    long my_hi = my_lo + my_rows;

    // Local buffers for the vectors. 
    // Each rank owns rows [my_lo, my_hi).
    // To compute A*p, we need the row above my_lo and the row below my_hi-1.
    double *local_r = (double *)malloc(my_rows * m * sizeof(double));
    double *local_p = (double *)malloc(my_rows * m * sizeof(double));
    double *local_q = (double *)malloc(my_rows * m * sizeof(double));
    double *local_x = (double *)malloc(my_rows * m * sizeof(double));
    
    // Halo buffers for p
    double *p_up = (double *)malloc(m * sizeof(double));
    double *p_down = (double *)malloc(m * sizeof(double));

    // Initialize x = 0, r = b, p = r
    memset(local_x, 0, my_rows * m * sizeof(double));
    for (long i = 0; i < my_rows; i++) {
        for (long j = 0; j < m; j++) {
            local_r[i * m + j] = b[(my_lo + i) * m + j];
            local_p[i * m + j] = local_r[i * m + j];
        }
    }

    double local_rr = 0.0;
    for (long i = 0; i < my_rows * m; i++) {
        local_rr += local_r[i] * local_r[i];
    }
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iters; iter++) {
        // Halo exchange for p
        MPI_Sendrecv(local_p, m, MPI_DOUBLE, up_rank, 0, 
                     p_down, m, MPI_DOUBLE, down_rank, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(local_p + (my_rows - 1) * m, m, MPI_DOUBLE, down_rank, 1, 
                     p_up, m, MPI_DOUBLE, up_rank, 1, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // q = A * p
        for (long i = 0; i < my_rows; i++) {
            long global_i = my_lo + i;
            for (long j = 0; j < m; j++) {
                double val = 4.0 * local_p[i * m + j];
                if (global_i > 0) {
                    val -= (i == 0) ? p_up[j] : local_p[(i - 1) * m + j];
                }
                if (global_i < m - 1) {
                    val -= (i == my_rows - 1) ? p_down[j] : local_p[(i + 1) * m + j];
                }
                if (j > 0) val -= local_p[i * m + (j - 1)];
                if (j < m - 1) val -= local_p[i * m + (j + 1)];
                local_q[i * m + j] = val;
            }
        }

        double local_pq = 0.0;
        for (long i = 0; i < my_rows * m; i++) {
            local_pq += local_p[i] * local_q[i];
        }
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rr / pq;
        for (long i = 0; i < my_rows * m; i++) {
            local_x[i] += alpha * local_p[i];
            local_r[i] -= alpha * local_q[i];
        }

        double local_rr_new = 0.0;
        for (long i = 0; i < my_rows * m; i++) {
            local_rr_new += local_r[i] * local_r[i];
        }
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;
        for (long i = 0; i < my_rows * m; i++) {
            local_p[i] = local_r[i] + beta * local_p[i];
        }
    }

    // Assemble full x
    long *counts = (long *)malloc(size * sizeof(long));
    long *displs = (long *)malloc(size * sizeof(long));
    for (int i = 0; i < size; i++) {
        long r_rows = m / size + (i < m % size ? 1 : 0);
        counts[i] = r_rows * m;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    MPI_Allgatherv(local_x, my_rows * m, MPI_DOUBLE, x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_r);
    free(local_p);
    free(local_q);
    free(local_x);
    free(p_up);
    free(p_down);
    free(counts);
    free(displs);
}
