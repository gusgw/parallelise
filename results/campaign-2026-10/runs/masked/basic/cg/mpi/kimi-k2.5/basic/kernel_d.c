#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kernel_d.h"

/* Apply 5-point stencil: 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1]
 * Each rank works on rows [lo, hi) of the m×m grid.
 * Halo exchange is done before calling this function.
 */
static void apply_A_local(double *q_local, const double *v_local, 
                          const double *halo_top, const double *halo_bottom,
                          long m, long lo, long hi)
{
    long nlocal = hi - lo;
    for (long i = 0; i < nlocal; i++) {
        long global_i = lo + i;
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            long v_idx = (i + 1) * m + j;  /* +1 because of halo row at top */
            double s = 4.0 * v_local[v_idx];
            
            /* Up neighbor */
            if (global_i > 0) {
                if (i == 0) {
                    s -= halo_top[j];
                } else {
                    s -= v_local[(i) * m + j];
                }
            }
            
            /* Down neighbor */
            if (global_i < m - 1) {
                if (i == nlocal - 1) {
                    s -= halo_bottom[j];
                } else {
                    s -= v_local[(i + 2) * m + j];
                }
            }
            
            /* Left neighbor */
            if (j > 0) {
                s -= v_local[v_idx - 1];
            }
            
            /* Right neighbor */
            if (j < m - 1) {
                s -= v_local[v_idx + 1];
            }
            
            q_local[idx] = s;
        }
    }
}

/* Local dot product for rows [lo, hi) */
static double dot_local(const double *u, const double *v, long m, long lo, long hi)
{
    double s = 0.0;
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            s += u[idx] * v[idx];
        }
    }
    return s;
}

/* Local vector operations for rows [lo, hi) */
static void x_plus_alpha_p(double *x, const double *p, double alpha, long m, long lo, long hi)
{
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            x[idx] += alpha * p[idx];
        }
    }
}

static void r_plus_beta_p(double *p, const double *r, double beta, long m, long lo, long hi)
{
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            p[idx] = r[idx] + beta * p[idx];
        }
    }
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Determine local row range [lo, hi) */
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long nlocal = hi - lo;
    
    /* Allocate local arrays with halo rows for p */
    /* v_local has (nlocal + 2) rows: halo top, local rows, halo bottom */
    double *v_local = malloc(sizeof(double) * (nlocal + 2) * m);
    double *q_local = malloc(sizeof(double) * nlocal * m);
    double *halo_send_top = malloc(sizeof(double) * m);
    double *halo_send_bottom = malloc(sizeof(double) * m);
    double *halo_recv_top = malloc(sizeof(double) * m);
    double *halo_recv_bottom = malloc(sizeof(double) * m);
    
    if (!v_local || !q_local || !halo_send_top || !halo_send_bottom ||
        !halo_recv_top || !halo_recv_bottom) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Initialize x = 0, r = b, p = r */
    /* x and r are full vectors (all ranks have full copy or we gather) */
    /* For efficiency, each rank stores full x, r, p but only updates its portion */
    
    double *r = malloc(sizeof(double) * m * m);
    double *p = malloc(sizeof(double) * m * m);
    
    if (!r || !p) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Initialize: x = 0, r = b, p = r */
    for (long i = 0; i < m * m; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }
    
    /* rr = r·r */
    double rr_local = dot_local(r, r, m, lo, hi);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iters; iter++) {
        /* Pack p into v_local with halo */
        /* v_local[0] is halo top, v_local[1..nlocal] are local rows, v_local[nlocal+1] is halo bottom */
        for (long i = 0; i < nlocal; i++) {
            for (long j = 0; j < m; j++) {
                v_local[(i + 1) * m + j] = p[(lo + i) * m + j];
            }
        }
        
        /* Pack halo send buffers */
        for (long j = 0; j < m; j++) {
            halo_send_top[j] = p[lo * m + j];
            halo_send_bottom[j] = p[(hi - 1) * m + j];
        }
        
        /* Halo exchange */
        int rank_up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int rank_down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
        
        /* Send to up, receive from down (for bottom halo) */
        MPI_Sendrecv(halo_send_top, m, MPI_DOUBLE, rank_up, 0,
                     halo_recv_bottom, m, MPI_DOUBLE, rank_down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send to down, receive from up (for top halo) */
        MPI_Sendrecv(halo_send_bottom, m, MPI_DOUBLE, rank_down, 1,
                     halo_recv_top, m, MPI_DOUBLE, rank_up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Copy received halos to v_local */
        if (rank > 0) {
            for (long j = 0; j < m; j++) {
                v_local[j] = halo_recv_top[j];
            }
        }
        if (rank < size - 1) {
            for (long j = 0; j < m; j++) {
                v_local[(nlocal + 1) * m + j] = halo_recv_bottom[j];
            }
        }
        
        /* q = A * p (local portion) */
        apply_A_local(q_local, v_local, halo_recv_top, halo_recv_bottom, m, lo, hi);
        
        /* p_dot_q = p·q (global) */
        double p_dot_q_local = 0.0;
        for (long i = lo; i < hi; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                long local_idx = (i - lo) * m + j;
                p_dot_q_local += p[idx] * q_local[local_idx];
            }
        }
        double p_dot_q;
        MPI_Allreduce(&p_dot_q_local, &p_dot_q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* alpha = rr / p_dot_q */
        double alpha = rr / p_dot_q;
        
        /* x = x + alpha * p */
        x_plus_alpha_p(x, p, alpha, m, lo, hi);
        
        /* r = r - alpha * q */
        for (long i = lo; i < hi; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                long local_idx = (i - lo) * m + j;
                r[idx] -= alpha * q_local[local_idx];
            }
        }
        
        /* rr_new = r·r */
        double rr_new_local = dot_local(r, r, m, lo, hi);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        
        /* p = r + beta * p */
        r_plus_beta_p(p, r, beta, m, lo, hi);
        
        rr = rr_new;
    }
    
    /* Gather full x to all ranks */
    /* Each rank has computed its portion, now we need all portions on all ranks */
    /* Use MPI_Allgatherv to handle non-uniform distribution */
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int my_count = (int)(nlocal * m);
    MPI_Allgather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    /* Compute displacements */
    for (int i = 0; i < size; i++) {
        long l = m * i / size;
        displs[i] = (int)(l * m);
    }
    
    /* Create temporary buffer for gathering */
    double *x_local = malloc(sizeof(double) * nlocal * m);
    if (!x_local) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Copy local portion to x_local */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            x_local[(i - lo) * m + j] = x[i * m + j];
        }
    }
    
    MPI_Allgatherv(x_local, (int)(nlocal * m), MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    free(x_local);
    free(recvcounts);
    free(displs);
    free(r);
    free(p);
    free(v_local);
    free(q_local);
    free(halo_send_top);
    free(halo_send_bottom);
    free(halo_recv_top);
    free(halo_recv_bottom);
}
