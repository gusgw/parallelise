#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major, index
 * idx = i * m + j), matrix-free, run for a FIXED number of iterations (no convergence
 * test -- the run must be deterministic and every rank must execute the same
 * collectives the same number of times). The operator is the 5-point stencil with
 * Dirichlet boundaries OUTSIDE the grid, so a neighbour off the edge contributes zero:
 *     (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1].
 * Standard CG from x = 0:  r = b; p = r; rr = dot(r,r);  then `iters` times:
 *     q = A p;  alpha = rr / dot(p,q);  x += alpha p;  r -= alpha q;
 *     rr_new = dot(r,r);  beta = rr_new / rr;  rr = rr_new;  p = r + beta p.
 * On return x holds the iterate after `iters` iterations.
 *
 * Implement THIS EXACT signature in cg.c. It is a COLLECTIVE routine: every MPI rank
 * calls it together with identical b (each rank holds a full copy), and on return
 * every rank holds the full solution x. Assume MPI is already initialised by the
 * caller (main); do NOT call MPI_Init or MPI_Finalize inside it. Decompose the grid
 * rows across the ranks: each rank owns rows [lo, hi) of x, r, p and q and updates
 * only those. Before EACH application of A, exchange one boundary row of p up and one
 * down with the neighbouring ranks (MPI_Sendrecv, with MPI_PROC_NULL at the edges so
 * the ends need no special cases). Compute the two dot products from local partial
 * sums combined with MPI_Allreduce (every rank needs alpha and beta). Assemble the
 * full x on every rank with a single MPI_Allgatherv at the end. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Decompose grid rows across ranks: each rank owns rows [lo_row, hi_row) */
    long lo_row = (m * rank) / size;
    long hi_row = (m * (rank + 1)) / size;
    long local_rows = hi_row - lo_row;
    long local_n = local_rows * m;
    
    /* Allocate local arrays for the rows this rank owns */
    double *x_local = (double *)malloc(local_n * sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));
    
    /* Buffers for halo exchange (one row above and one row below) */
    double *p_above = (double *)malloc(m * sizeof(double));
    double *p_below = (double *)malloc(m * sizeof(double));
    
    /* Initialize x = 0, r = b, p = r for local rows */
    for (long k = 0; k < local_n; k++) {
        x_local[k] = 0.0;
        r_local[k] = b[lo_row * m + k];
        p_local[k] = r_local[k];
    }
    
    /* Compute initial rr = dot(r, r) from local partial sum */
    double rr_local = 0.0;
    for (long k = 0; k < local_n; k++) {
        rr_local += r_local[k] * r_local[k];
    }
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    /* Determine neighbors for halo exchange */
    int rank_above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    /* CG iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows of p with neighbors */
        /* Send first local row up, receive into p_above */
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, rank_above, 0,
                     p_above, m, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send last local row down, receive into p_below */
        MPI_Sendrecv(&p_local[(local_rows - 1) * m], m, MPI_DOUBLE, rank_below, 0,
                     p_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Apply A to p: q = A p (only for local rows) */
        for (long i = 0; i < local_rows; i++) {
            long global_i = lo_row + i;
            for (long j = 0; j < m; j++) {
                long local_idx = i * m + j;
                double s = 4.0 * p_local[local_idx];
                
                /* Neighbor above (i-1) */
                if (global_i > 0) {
                    if (i > 0) {
                        s -= p_local[(i - 1) * m + j];
                    } else {
                        s -= p_above[j];
                    }
                }
                
                /* Neighbor below (i+1) */
                if (global_i < m - 1) {
                    if (i < local_rows - 1) {
                        s -= p_local[(i + 1) * m + j];
                    } else {
                        s -= p_below[j];
                    }
                }
                
                /* Neighbor left (j-1) */
                if (j > 0) {
                    s -= p_local[local_idx - 1];
                }
                
                /* Neighbor right (j+1) */
                if (j < m - 1) {
                    s -= p_local[local_idx + 1];
                }
                
                q_local[local_idx] = s;
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
    
    /* Gather full solution x on all ranks using MPI_Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long r_lo_row = (m * r) / size;
        long r_hi_row = (m * (r + 1)) / size;
        recvcounts[r] = (int)((r_hi_row - r_lo_row) * m);
        displs[r] = (int)(r_lo_row * m);
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
