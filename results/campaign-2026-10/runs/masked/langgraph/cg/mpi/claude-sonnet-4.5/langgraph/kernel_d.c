#include "kernel_d.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Distribute rows across ranks
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;
    
    // Allocate local arrays
    double *x_local = malloc(sizeof(double) * local_n);
    double *r_local = malloc(sizeof(double) * local_n);
    double *p_local = malloc(sizeof(double) * local_n);
    double *q_local = malloc(sizeof(double) * local_n);
    
    // Halo buffers
    double *top_halo = malloc(sizeof(double) * m);
    double *bot_halo = malloc(sizeof(double) * m);
    
    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Initialize local portions: x = 0, r = b, p = r
    for (long i = 0; i < local_n; i++) {
        x_local[i] = 0.0;
        r_local[i] = b[lo * m + i];
        p_local[i] = r_local[i];
    }
    
    // Compute initial rr
    double local_rr = 0.0;
    for (long i = 0; i < local_n; i++) {
        local_rr += r_local[i] * r_local[i];
    }
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iters; iter++) {
        // Exchange halos for p
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, prev_rank, 0,
                     bot_halo, m, MPI_DOUBLE, next_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&p_local[(local_rows - 1) * m], m, MPI_DOUBLE, next_rank, 1,
                     top_halo, m, MPI_DOUBLE, prev_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Apply A to local rows
        for (long i = 0; i < local_rows; i++) {
            long global_i = lo + i;
            for (long j = 0; j < m; j++) {
                long local_idx = i * m + j;
                double s = 4.0 * p_local[local_idx];
                
                // Top neighbor
                if (global_i > 0) {
                    if (i == 0) {
                        s -= top_halo[j];
                    } else {
                        s -= p_local[local_idx - m];
                    }
                }
                
                // Bottom neighbor
                if (global_i < m - 1) {
                    if (i == local_rows - 1) {
                        s -= bot_halo[j];
                    } else {
                        s -= p_local[local_idx + m];
                    }
                }
                
                // Left neighbor
                if (j > 0) {
                    s -= p_local[local_idx - 1];
                }
                
                // Right neighbor
                if (j < m - 1) {
                    s -= p_local[local_idx + 1];
                }
                
                q_local[local_idx] = s;
            }
        }
        
        // Compute local dot product p . q
        double local_pq = 0.0;
        for (long i = 0; i < local_n; i++) {
            local_pq += p_local[i] * q_local[i];
        }
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double alpha = rr / pq;
        
        // Update x and r locally
        for (long i = 0; i < local_n; i++) {
            x_local[i] += alpha * p_local[i];
            r_local[i] -= alpha * q_local[i];
        }
        
        // Compute new rr
        double local_rr_new = 0.0;
        for (long i = 0; i < local_n; i++) {
            local_rr_new += r_local[i] * r_local[i];
        }
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double beta = rr_new / rr;
        rr = rr_new;
        
        // Update p locally
        for (long i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }
    
    // Gather results to all ranks
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long r_lo = m * r / size;
        long r_hi = m * (r + 1) / size;
        recvcounts[r] = (r_hi - r_lo) * m;
        displs[r] = r_lo * m;
    }
    
    MPI_Allgatherv(x_local, local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    free(recvcounts);
    free(displs);
    free(top_halo);
    free(bot_halo);
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
}
