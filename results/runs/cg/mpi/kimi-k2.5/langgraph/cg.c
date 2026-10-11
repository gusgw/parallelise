#include "cg.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: apply 5-point Poisson stencil to local rows of p.
 * Each rank owns rows [lo, hi) of the m x m grid.
 * Before calling this, p must have its halo rows exchanged. */
static void apply_A_local(double *q_local, const double *p_with_halo,
                          long m, long lo, long hi)
{
    /* p_with_halo contains rows [lo-1, hi+1) if they exist.
     * Local rows are at offset 1 in p_with_halo (unless lo==0, then offset 0). */
    long local_rows = hi - lo;
    for (long i = 0; i < local_rows; i++) {
        long global_i = lo + i;
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            /* p at current cell: account for halo offset */
            const double *p_row = p_with_halo + (i + (lo > 0 ? 1 : 0)) * m;
            double s = 4.0 * p_row[j];
            /* up neighbor */
            if (global_i > 0) {
                const double *p_up = p_with_halo + (i + (lo > 0 ? 1 : 0) - 1) * m;
                s -= p_up[j];
            }
            /* down neighbor */
            if (global_i < m - 1) {
                const double *p_down = p_with_halo + (i + (lo > 0 ? 1 : 0) + 1) * m;
                s -= p_down[j];
            }
            /* left neighbor */
            if (j > 0) {
                s -= p_row[j - 1];
            }
            /* right neighbor */
            if (j < m - 1) {
                s -= p_row[j + 1];
            }
            q_local[idx] = s;
        }
    }
}

/* Helper: compute dot product of two local vectors */
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

    /* Decompose rows across ranks: each rank owns rows [lo, hi) */
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m;

    /* Allocate local arrays */
    double *x_local = (double *)malloc(local_n * sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));
    
    /* p_with_halo has space for up to 2 extra rows (one above, one below) */
    long halo_rows = local_rows + (lo > 0 ? 1 : 0) + (hi < m ? 1 : 0);
    double *p_with_halo = (double *)malloc(halo_rows * m * sizeof(double));

    /* Initialize: x = 0, r = b, p = r */
    for (long i = 0; i < local_n; i++) {
        x_local[i] = 0.0;
        /* b is full on every rank, extract our portion */
        long global_idx = lo * m + i;
        r_local[i] = b[global_idx];
        p_local[i] = r_local[i];
    }

    /* Compute initial rr = dot(r, r) */
    double rr_local = dot_local(r_local, r_local, local_n);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Determine neighbors for halo exchange */
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows of p before applying A */
        /* Copy local p into p_with_halo at the right offset */
        long p_offset = (lo > 0 ? 1 : 0);
        memcpy(p_with_halo + p_offset * m, p_local, local_n * sizeof(double));

        /* Send top row up, receive bottom row from up */
        if (lo > 0) {
            MPI_Sendrecv(p_local, m, MPI_DOUBLE, up, 0,
                         p_with_halo, m, MPI_DOUBLE, up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /* Send bottom row down, receive top row from down */
        if (hi < m) {
            MPI_Sendrecv(p_local + (local_rows - 1) * m, m, MPI_DOUBLE, down, 1,
                         p_with_halo + (halo_rows - 1) * m, m, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* q = A * p */
        apply_A_local(q_local, p_with_halo, m, lo, hi);

        /* alpha = rr / dot(p, q) */
        double pdotq_local = dot_local(p_local, q_local, local_n);
        double pdotq;
        MPI_Allreduce(&pdotq_local, &pdotq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pdotq;

        /* x += alpha * p */
        for (long i = 0; i < local_n; i++) {
            x_local[i] += alpha * p_local[i];
        }

        /* r -= alpha * q */
        for (long i = 0; i < local_n; i++) {
            r_local[i] -= alpha * q_local[i];
        }

        /* rr_new = dot(r, r) */
        double rr_new_local = dot_local(r_local, r_local, local_n);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        for (long i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }
    }

    /* Gather full x on all ranks using Allgatherv */
    /* First, compute counts and displacements */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    for (int r = 0; r < size; r++) {
        long r_lo = m * r / size;
        long r_hi = m * (r + 1) / size;
        long r_rows = r_hi - r_lo;
        recvcounts[r] = (int)(r_rows * m);
        displs[r] = (int)(r_lo * m);
    }

    MPI_Allgatherv(x_local, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Cleanup */
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(p_with_halo);
    free(recvcounts);
    free(displs);
}
