#include "cg.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations.
 * The operator is the 5-point stencil with Dirichlet boundaries OUTSIDE the grid.
 * Standard CG from x = 0.
 * On return x holds the iterate after `iters` iterations on all ranks. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose rows across ranks: each rank owns rows [lo, hi) */
    long lo = (long)rank * m / size;
    long hi = (long)(rank + 1) * m / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;

    /* Allocate local arrays for x, r, p, q */
    double *local_x = (double *)malloc(local_n * sizeof(double));
    double *local_r = (double *)malloc(local_n * sizeof(double));
    double *local_p = (double *)malloc(local_n * sizeof(double));
    double *local_q = (double *)malloc(local_n * sizeof(double));

    /* Initialize x = 0, r = b, p = r */
    for (long i = 0; i < local_n; i++) {
        local_x[i] = 0.0;
        long global_idx = lo * m + i;
        local_r[i] = b[global_idx];
        local_p[i] = local_r[i];
    }

    /* Compute initial rr = dot(r, r) */
    double local_rr = 0.0;
    for (long i = 0; i < local_n; i++) {
        local_rr += local_r[i] * local_r[i];
    }
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Allocate halo buffers for p (one row above, one row below) */
    double *p_above = (double *)malloc(m * sizeof(double));
    double *p_below = (double *)malloc(m * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange boundary rows of p with neighbors */
        /* Send row lo to rank-1 (above), receive row hi from rank+1 (below) */
        /* Send row hi-1 to rank+1 (below), receive row lo from rank-1 (above) */

        /* Determine neighbors */
        int rank_above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        /* Exchange: send first row up, receive last row from below */
        MPI_Sendrecv(local_p, m, MPI_DOUBLE, rank_above, 0,
                     p_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Exchange: send last row down, receive first row from above */
        MPI_Sendrecv(local_p + (local_rows - 1) * m, m, MPI_DOUBLE, rank_below, 1,
                     p_above, m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p: q = A * p (local computation) */
        for (long i = lo; i < hi; i++) {
            long local_i = i - lo;
            for (long j = 0; j < m; j++) {
                long local_idx = local_i * m + j;
                double val = 4.0 * local_p[local_idx];

                /* Up neighbor */
                if (i > 0) {
                    if (local_i > 0) {
                        val -= local_p[(local_i - 1) * m + j];
                    } else {
                        val -= p_above[j];
                    }
                }

                /* Down neighbor */
                if (i < m - 1) {
                    if (local_i < local_rows - 1) {
                        val -= local_p[(local_i + 1) * m + j];
                    } else {
                        val -= p_below[j];
                    }
                }

                /* Left neighbor */
                if (j > 0) {
                    val -= local_p[local_i * m + (j - 1)];
                }

                /* Right neighbor */
                if (j < m - 1) {
                    val -= local_p[local_i * m + (j + 1)];
                }

                local_q[local_idx] = val;
            }
        }

        /* Compute dot(p, q) */
        double local_pq = 0.0;
        for (long i = 0; i < local_n; i++) {
            local_pq += local_p[i] * local_q[i];
        }
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* alpha = rr / dot(p, q) */
        double alpha = rr / pq;

        /* x += alpha * p */
        /* r -= alpha * q */
        for (long i = 0; i < local_n; i++) {
            local_x[i] += alpha * local_p[i];
            local_r[i] -= alpha * local_q[i];
        }

        /* Compute rr_new = dot(r, r) */
        double local_rr_new = 0.0;
        for (long i = 0; i < local_n; i++) {
            local_rr_new += local_r[i] * local_r[i];
        }
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;

        /* Update rr */
        rr = rr_new;

        /* p = r + beta * p */
        for (long i = 0; i < local_n; i++) {
            local_p[i] = local_r[i] + beta * local_p[i];
        }
    }

    /* Assemble full x on all ranks using MPI_Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        long r_lo = (long)r * m / size;
        long r_hi = (long)(r + 1) * m / size;
        long r_rows = r_hi - r_lo;
        recvcounts[r] = (int)(r_rows * m);
    }

    displs[0] = 0;
    for (int r = 1; r < size; r++) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
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
