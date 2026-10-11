#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid.
 * Parallel implementation using MPI with row decomposition.
 * Each rank owns rows [lo, hi) and exchanges boundary rows with neighbors.
 */
void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Decompose rows across ranks: each rank owns rows [lo, hi) */
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;

    /* Allocate local vectors for owned rows */
    double *x_local = (double *)malloc(local_n * sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));
    
    /* Buffers for boundary rows from neighbors */
    double *p_above = (double *)malloc(m * sizeof(double));
    double *p_below = (double *)malloc(m * sizeof(double));

    /* Initialize x = 0, r = b, p = r for local rows */
    for (long k = 0; k < local_n; k++) {
        x_local[k] = 0.0;
        r_local[k] = b[lo * m + k];
        p_local[k] = r_local[k];
    }

    /* Compute initial rr = dot(r, r) */
    double rr_local = 0.0;
    for (long k = 0; k < local_n; k++) {
        rr_local += r_local[k] * r_local[k];
    }
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* CG iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange boundary rows of p with neighbors */
        int rank_above = (lo > 0) ? rank - 1 : MPI_PROC_NULL;
        int rank_below = (hi < m) ? rank + 1 : MPI_PROC_NULL;
        
        /* Send first row up, receive from above */
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, rank_above, 0,
                     p_above, m, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send last row down, receive from below */
        MPI_Sendrecv(&p_local[(local_rows - 1) * m], m, MPI_DOUBLE, rank_below, 0,
                     p_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p: q = A * p (5-point stencil) */
        for (long i = 0; i < local_rows; i++) {
            long global_i = lo + i;
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                double s = 4.0 * p_local[idx];
                
                /* Neighbor above (i-1) */
                if (global_i > 0) {
                    if (i > 0) {
                        s -= p_local[idx - m];
                    } else {
                        s -= p_above[j];
                    }
                }
                
                /* Neighbor below (i+1) */
                if (global_i < m - 1) {
                    if (i < local_rows - 1) {
                        s -= p_local[idx + m];
                    } else {
                        s -= p_below[j];
                    }
                }
                
                /* Neighbor left (j-1) */
                if (j > 0) {
                    s -= p_local[idx - 1];
                }
                
                /* Neighbor right (j+1) */
                if (j < m - 1) {
                    s -= p_local[idx + 1];
                }
                
                q_local[idx] = s;
            }
        }

        /* Compute alpha = rr / dot(p, q) */
        double pq_local = 0.0;
        for (long k = 0; k < local_n; k++) {
            pq_local += p_local[k] * q_local[k];
        }
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* x += alpha * p */
        for (long k = 0; k < local_n; k++) {
            x_local[k] += alpha * p_local[k];
        }

        /* r -= alpha * q */
        for (long k = 0; k < local_n; k++) {
            r_local[k] -= alpha * q_local[k];
        }

        /* Compute rr_new = dot(r, r) */
        double rr_new_local = 0.0;
        for (long k = 0; k < local_n; k++) {
            rr_new_local += r_local[k] * r_local[k];
        }
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        for (long k = 0; k < local_n; k++) {
            p_local[k] = r_local[k] + beta * p_local[k];
        }
    }

    /* Gather full solution x on all ranks using Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    for (int r = 0; r < size; r++) {
        long r_lo = m * r / size;
        long r_hi = m * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * m);
        displs[r] = (int)(r_lo * m);
    }
    
    MPI_Allgatherv(x_local, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Clean up */
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(p_above);
    free(p_below);
    free(recvcounts);
    free(displs);
}
