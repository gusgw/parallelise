#include "kernel_d.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

static double dot_local(const double *u, const double *v, long n_local)
{
    double s = 0.0;
    for (long k = 0; k < n_local; k++) {
        s += u[k] * v[k];
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose rows across ranks */
    long row_start = (m * rank) / size;
    long row_end = (m * (rank + 1)) / size;
    long n_local = (row_end - row_start) * m;

    /* Allocate local arrays */
    double *x_local = malloc(sizeof(double) * n_local);
    double *r_local = malloc(sizeof(double) * n_local);
    double *p_local = malloc(sizeof(double) * n_local);
    double *q_local = malloc(sizeof(double) * n_local);
    double *b_local = malloc(sizeof(double) * n_local);
    
    /* Halo buffers for apply_A */
    double *p_up = malloc(sizeof(double) * m);
    double *p_down = malloc(sizeof(double) * m);
    double *recv_up = malloc(sizeof(double) * m);
    double *recv_down = malloc(sizeof(double) * m);

    if (!x_local || !r_local || !p_local || !q_local || !b_local ||
        !p_up || !p_down || !recv_up || !recv_down) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Copy local portion of b */
    for (long i = row_start; i < row_end; i++) {
        for (long j = 0; j < m; j++) {
            long global_idx = i * m + j;
            long local_idx = (i - row_start) * m + j;
            b_local[local_idx] = b[global_idx];
        }
    }

    /* Initialize: x = 0, r = b, p = r */
    for (long k = 0; k < n_local; k++) {
        x_local[k] = 0.0;
        r_local[k] = b_local[k];
        p_local[k] = r_local[k];
    }

    /* rr = dot(r, r) */
    double rr_local = dot_local(r_local, r_local, n_local);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo for p */
        /* Pack top and bottom rows of p_local */
        if (n_local > 0) {
            memcpy(p_up, p_local, sizeof(double) * m);
            memcpy(p_down, p_local + (n_local - m), sizeof(double) * m);
        }

        /* Send to up neighbor, receive from down neighbor */
        int rank_up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int rank_down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        MPI_Sendrecv(p_up, (int)m, MPI_DOUBLE, rank_up, 0,
                     recv_down, (int)m, MPI_DOUBLE, rank_down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send to down neighbor, receive from up neighbor */
        MPI_Sendrecv(p_down, (int)m, MPI_DOUBLE, rank_down, 1,
                     recv_up, (int)m, MPI_DOUBLE, rank_up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p_local */
        for (long i = row_start; i < row_end; i++) {
            for (long j = 0; j < m; j++) {
                long local_idx = (i - row_start) * m + j;
                double s = 4.0 * p_local[local_idx];
                
                /* Up neighbor */
                if (i > 0) {
                    if (i == row_start) {
                        if (rank > 0) {
                            s -= recv_up[j];
                        }
                    } else {
                        s -= p_local[local_idx - m];
                    }
                }
                
                /* Down neighbor */
                if (i < m - 1) {
                    if (i == row_end - 1) {
                        if (rank < size - 1) {
                            s -= recv_down[j];
                        }
                    } else {
                        s -= p_local[local_idx + m];
                    }
                }
                
                /* Left neighbor */
                if (j > 0) {
                    s -= p_local[local_idx - 1];
                }
                
                /* Right neighbor */
                if (j < m - 1) {
                    s -= p_local[local_idx + 1];
                }
                
                q_local[local_idx] = s;
            }
        }

        /* dot(p, q) */
        double pdotq_local = dot_local(p_local, q_local, n_local);
        double pdotq;
        MPI_Allreduce(&pdotq_local, &pdotq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* alpha = rr / dot(p, q) */
        double alpha = rr / pdotq;

        /* x += alpha * p */
        for (long k = 0; k < n_local; k++) {
            x_local[k] += alpha * p_local[k];
        }

        /* r -= alpha * q */
        for (long k = 0; k < n_local; k++) {
            r_local[k] -= alpha * q_local[k];
        }

        /* rr_new = dot(r, r) */
        double rr_new_local = dot_local(r_local, r_local, n_local);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        for (long k = 0; k < n_local; k++) {
            p_local[k] = r_local[k] + beta * p_local[k];
        }
    }

    /* Gather results to all ranks */
    /* Each rank has n_local elements, need to gather to x */
    /* Use MPI_Allgatherv - all ranks need recvcounts and displs */
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int my_count = (int)n_local;
    MPI_Allgather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    MPI_Allgatherv(x_local, (int)n_local, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Cleanup */
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(b_local);
    free(p_up);
    free(p_down);
    free(recv_up);
    free(recv_down);
    free(recvcounts);
    free(displs);
}
