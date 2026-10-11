#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kernel_d.h"

/* Apply 5-point stencil to local rows with halo exchange.
 * Each rank owns rows [row_lo, row_hi).
 * Input p includes halo rows at top and bottom.
 * Output q is only for owned rows.
 */
static void apply_A_local(double *q, const double *p, long m, long row_lo, long row_hi)
{
    long nlocal = row_hi - row_lo;
    
    for (long i = 0; i < nlocal; i++) {
        long global_i = row_lo + i;
        for (long j = 0; j < m; j++) {
            long idx_local = i * m + j;
            long idx_halo = (i + 1) * m + j;  /* +1 for top halo row */
            
            double s = 4.0 * p[idx_halo];
            
            /* Up neighbor */
            if (global_i > 0) {
                s -= p[idx_halo - m];
            }
            /* Down neighbor */
            if (global_i < m - 1) {
                s -= p[idx_halo + m];
            }
            /* Left neighbor */
            if (j > 0) {
                s -= p[idx_halo - 1];
            }
            /* Right neighbor */
            if (j < m - 1) {
                s -= p[idx_halo + 1];
            }
            
            q[idx_local] = s;
        }
    }
}

/* Compute dot product of local portions */
static double dot_local(const double *u, const double *v, long n)
{
    double s = 0.0;
    for (long i = 0; i < n; i++) {
        s += u[i] * v[i];
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long n = m * m;
    
    /* Determine row range for this rank */
    long row_lo = m * rank / size;
    long row_hi = m * (rank + 1) / size;
    long nlocal = row_hi - row_lo;
    long local_size = nlocal * m;
    
    /* Allocate local arrays */
    /* x_local, r_local, p_local are for owned rows only */
    double *x_local = calloc(local_size, sizeof(double));
    double *r_local = malloc(local_size * sizeof(double));
    double *p_local = malloc(local_size * sizeof(double));
    double *q_local = malloc(local_size * sizeof(double));
    
    /* p_halo includes one row above and one row below for stencil */
    /* Size: (nlocal + 2) * m */
    double *p_halo = malloc((nlocal + 2) * m * sizeof(double));
    
    if (!x_local || !r_local || !p_local || !q_local || !p_halo) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Initialize r = b (local portion) */
    for (long i = 0; i < local_size; i++) {
        r_local[i] = b[row_lo * m + i];
    }
    
    /* p = r */
    memcpy(p_local, r_local, local_size * sizeof(double));
    
    /* rr = r^T * r */
    double rr_local = dot_local(r_local, r_local, local_size);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iters; iter++) {
        /* Fill p_halo with local p values */
        memcpy(p_halo + m, p_local, local_size * sizeof(double));
        
        /* Halo exchange */
        MPI_Status status;
        
        /* Send to top, receive from bottom */
        int top_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int bot_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
        
        /* Send top row to rank-1, receive bottom halo from rank+1 */
        MPI_Sendrecv(p_halo + m, m, MPI_DOUBLE, top_rank, 0,
                     p_halo + (nlocal + 1) * m, m, MPI_DOUBLE, bot_rank, 0,
                     MPI_COMM_WORLD, &status);
        
        /* Send bottom row to rank+1, receive top halo from rank-1 */
        MPI_Sendrecv(p_halo + nlocal * m, m, MPI_DOUBLE, bot_rank, 1,
                     p_halo, m, MPI_DOUBLE, top_rank, 1,
                     MPI_COMM_WORLD, &status);
        
        /* q = A * p */
        apply_A_local(q_local, p_halo, m, row_lo, row_hi);
        
        /* p_dot_q = p^T * q */
        double pdq_local = dot_local(p_local, q_local, local_size);
        double pdq;
        MPI_Allreduce(&pdq_local, &pdq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* alpha = rr / pdq */
        double alpha = rr / pdq;
        
        /* x = x + alpha * p */
        for (long i = 0; i < local_size; i++) {
            x_local[i] += alpha * p_local[i];
        }
        
        /* r = r - alpha * q */
        for (long i = 0; i < local_size; i++) {
            r_local[i] -= alpha * q_local[i];
        }
        
        /* rr_new = r^T * r */
        double rr_new_local = dot_local(r_local, r_local, local_size);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;
        
        /* p = r + beta * p */
        for (long i = 0; i < local_size; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }
    
    /* Gather x from all ranks to rank 0, then broadcast to all */
    /* First, gather counts */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    
    int mycount = (int)local_size;
    MPI_Gather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(x_local, mycount, MPI_DOUBLE,
                x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Broadcast full x to all ranks */
    MPI_Bcast(x, (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Cleanup */
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(p_halo);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}
