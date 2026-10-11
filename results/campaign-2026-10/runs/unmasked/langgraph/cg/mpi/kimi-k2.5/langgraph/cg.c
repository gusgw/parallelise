#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"

/* Apply the 5-point Poisson stencil to p locally on rows [lo, hi).
 * p_full contains the local rows plus halo rows (if any).
 * The local rows of p are stored in p_full at offset 1 (if lo > 0) or 0 (if lo == 0).
 * q is the output for rows [lo, hi). */
static void apply_A_local(double *q, const double *p_full, long m, long lo, long hi)
{
    long p_offset = (lo > 0) ? 1 : 0;
    
    for (long i = lo; i < hi; i++) {
        long local_i = i - lo;
        long p_row_offset = (local_i + p_offset) * m;
        for (long j = 0; j < m; j++) {
            long idx = local_i * m + j;
            double s = 4.0 * p_full[p_row_offset + j];
            /* up neighbor */
            if (i > 0) {
                s -= p_full[(local_i + p_offset - 1) * m + j];
            }
            /* down neighbor */
            if (i < m - 1) {
                s -= p_full[(local_i + p_offset + 1) * m + j];
            }
            /* left neighbor */
            if (j > 0) {
                s -= p_full[p_row_offset + j - 1];
            }
            /* right neighbor */
            if (j < m - 1) {
                s -= p_full[p_row_offset + j + 1];
            }
            q[idx] = s;
        }
    }
}

/* Compute local dot product of two vectors on rows [lo, hi). */
static double dot_local(const double *u, const double *v, long m, long lo, long hi)
{
    double s = 0.0;
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = (i - lo) * m + j;
            s += u[idx] * v[idx];
        }
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Determine row range for this rank */
    long lo = (long)rank * m / size;
    long hi = (long)(rank + 1) * m / size;
    long nrows = hi - lo;
    long local_n = nrows * m;
    
    /* Allocate local arrays */
    double *x_local = calloc(local_n, sizeof(double));  /* x starts at 0 */
    double *r_local = malloc(local_n * sizeof(double));
    double *p_local = malloc(local_n * sizeof(double));
    double *q_local = malloc(local_n * sizeof(double));
    
    /* p_full includes halo rows: one above (if not first rank) and one below (if not last rank) */
    long p_full_rows = nrows + (lo > 0 ? 1 : 0) + (hi < m ? 1 : 0);
    double *p_full = malloc(p_full_rows * m * sizeof(double));
    
    /* Initialize: x = 0, r = b, p = r */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long local_idx = (i - lo) * m + j;
            long global_idx = i * m + j;
            r_local[local_idx] = b[global_idx];
            p_local[local_idx] = r_local[local_idx];
        }
    }
    
    /* rr = dot(r, r) */
    double rr_local = dot_local(r_local, r_local, m, lo, hi);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange boundary rows of p before applying A */
        /* Copy local p into p_full at the right offset */
        long p_offset = (lo > 0) ? 1 : 0;
        memcpy(p_full + p_offset * m, p_local, local_n * sizeof(double));
        
        /* Send to upper neighbor (rank-1), receive from lower neighbor (rank+1) */
        double *recv_buf_down = (rank < size - 1) ? p_full + (p_offset + nrows) * m : p_full;
        MPI_Sendrecv(
            p_full + p_offset * m,           /* send buf: first local row */
            (int)m, MPI_DOUBLE,
            (rank > 0) ? rank - 1 : MPI_PROC_NULL, 0,
            recv_buf_down,  /* recv buf: after last local row, or dummy if no neighbor */
            (int)m, MPI_DOUBLE,
            (rank < size - 1) ? rank + 1 : MPI_PROC_NULL, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        
        /* Send to lower neighbor (rank+1), receive from upper neighbor (rank-1) */
        double *recv_buf_up = (rank > 0) ? p_full : p_full;
        MPI_Sendrecv(
            p_full + (p_offset + nrows - 1) * m,  /* send buf: last local row */
            (int)m, MPI_DOUBLE,
            (rank < size - 1) ? rank + 1 : MPI_PROC_NULL, 1,
            recv_buf_up,  /* recv buf: before first local row, or dummy if no neighbor */
            (int)m, MPI_DOUBLE,
            (rank > 0) ? rank - 1 : MPI_PROC_NULL, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        
        /* q = A * p */
        apply_A_local(q_local, p_full, m, lo, hi);
        
        /* pdotq = dot(p, q) */
        double pdotq_local = dot_local(p_local, q_local, m, lo, hi);
        double pdotq;
        MPI_Allreduce(&pdotq_local, &pdotq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* alpha = rr / pdotq */
        double alpha = rr / pdotq;
        
        /* x += alpha * p; r -= alpha * q */
        for (long k = 0; k < local_n; k++) {
            x_local[k] += alpha * p_local[k];
            r_local[k] -= alpha * q_local[k];
        }
        
        /* rr_new = dot(r, r) */
        double rr_new_local = dot_local(r_local, r_local, m, lo, hi);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* beta = rr_new / rr; rr = rr_new */
        double beta = rr_new / rr;
        rr = rr_new;
        
        /* p = r + beta * p */
        for (long k = 0; k < local_n; k++) {
            p_local[k] = r_local[k] + beta * p_local[k];
        }
    }
    
    /* Gather the full solution x on all ranks using MPI_Allgatherv */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    
    int my_count = (int)local_n;
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    /* Broadcast recvcounts and displs to all ranks for Allgatherv */
    if (rank != 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Allgatherv(x_local, my_count, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    free(recvcounts);
    free(displs);
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(p_full);
}
