#include "kernel_d.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

static void apply_A(double *q, const double *v, long m,
                    long lo_row, long hi_row, const double *halo_above, const double *halo_below)
{
    for (long i = lo_row; i < hi_row; i++) {
        for (long j = 0; j < m; j++) {
            long local_i = i - lo_row;
            long idx = local_i * m + j;
            double s = 4.0 * v[idx];
            
            // Above neighbor
            if (i > 0) {
                if (local_i > 0) {
                    s -= v[idx - m];
                } else {
                    s -= halo_above[j];
                }
            }
            
            // Below neighbor
            if (i < m - 1) {
                if (local_i < hi_row - lo_row - 1) {
                    s -= v[idx + m];
                } else {
                    s -= halo_below[j];
                }
            }
            
            // Left neighbor
            if (j > 0) {
                s -= v[idx - 1];
            }
            
            // Right neighbor
            if (j < m - 1) {
                s -= v[idx + 1];
            }
            
            q[idx] = s;
        }
    }
}

static double dot_local(const double *u, const double *v, long n)
{
    double s = 0.0;
    for (long k = 0; k < n; k++) {
        s += u[k] * v[k];
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Decompose rows across ranks
    long lo_row = (m * rank) / size;
    long hi_row = (m * (rank + 1)) / size;
    long local_rows = hi_row - lo_row;
    long local_n = local_rows * m;
    
    // Allocate local arrays
    double *local_x = malloc(sizeof(double) * local_n);
    double *local_r = malloc(sizeof(double) * local_n);
    double *local_p = malloc(sizeof(double) * local_n);
    double *local_q = malloc(sizeof(double) * local_n);
    double *local_b = malloc(sizeof(double) * local_n);
    double *halo_above = malloc(sizeof(double) * m);
    double *halo_below = malloc(sizeof(double) * m);
    
    // Initialize local portion of b
    for (long i = 0; i < local_rows; i++) {
        for (long j = 0; j < m; j++) {
            long global_idx = (lo_row + i) * m + j;
            long local_idx = i * m + j;
            local_b[local_idx] = b[global_idx];
        }
    }
    
    // Initialize x = 0, r = b, p = r
    for (long k = 0; k < local_n; k++) {
        local_x[k] = 0.0;
        local_r[k] = local_b[k];
        local_p[k] = local_r[k];
    }
    
    double rr_local = dot_local(local_r, local_r, local_n);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    int rank_above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    for (int iter = 0; iter < iters; iter++) {
        // Exchange halos for p
        MPI_Sendrecv(local_p, m, MPI_DOUBLE, rank_above, 0,
                     halo_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&local_p[(local_rows - 1) * m], m, MPI_DOUBLE, rank_below, 1,
                     halo_above, m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Apply A to p
        apply_A(local_q, local_p, m, lo_row, hi_row, halo_above, halo_below);
        
        // Compute alpha
        double pq_local = dot_local(local_p, local_q, local_n);
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;
        
        // Update x and r
        for (long k = 0; k < local_n; k++) {
            local_x[k] += alpha * local_p[k];
            local_r[k] -= alpha * local_q[k];
        }
        
        // Compute new rr
        double rr_new_local = dot_local(local_r, local_r, local_n);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double beta = rr_new / rr;
        rr = rr_new;
        
        // Update p
        for (long k = 0; k < local_n; k++) {
            local_p[k] = local_r[k] + beta * local_p[k];
        }
    }
    
    // Gather result to all ranks
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long r_lo = (m * r) / size;
        long r_hi = (m * (r + 1)) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * m);
        displs[r] = (int)(r_lo * m);
    }
    
    MPI_Allgatherv(local_x, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    free(local_x);
    free(local_r);
    free(local_p);
    free(local_q);
    free(local_b);
    free(halo_above);
    free(halo_below);
    free(recvcounts);
    free(displs);
}
