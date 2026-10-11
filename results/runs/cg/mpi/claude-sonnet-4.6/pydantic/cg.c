/*
 * cg.c  --  Parallel conjugate gradient for the 2-D Poisson system.
 *
 * Algorithm (from cg.h / cg.ipynb):
 *   Operator : (A v)[i,j] = 4*v[i,j] - v[i-1,j] - v[i+1,j] - v[i,j-1] - v[i,j+1]
 *              Dirichlet (zero) outside the m x m grid.
 *   CG from x = 0:
 *     r = b;  p = r;  rr = dot(r,r);
 *     for iters:
 *       q = A p
 *       alpha = rr / dot(p,q)
 *       x += alpha * p
 *       r -= alpha * q
 *       rr_new = dot(r,r)
 *       beta  = rr_new / rr
 *       rr    = rr_new
 *       p = r + beta * p
 *
 * Parallelism:
 *   - Grid rows are block-partitioned: rank r owns rows [lo, hi).
 *   - Before each A application, one halo row of p is exchanged up/down
 *     with MPI_Sendrecv (MPI_PROC_NULL at the boundary ranks).
 *   - Dot products are reduced with MPI_Allreduce (every rank needs alpha/beta).
 *   - At the end, the full x is assembled on every rank with MPI_Allgatherv.
 */

#include "cg.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

/* Local dot product over [0, len). */
static double local_dot(const double *u, const double *v, long len)
{
    double s = 0.0;
    for (long k = 0; k < len; k++) s += u[k] * v[k];
    return s;
}

/*
 * apply_A_local
 *
 * Compute q[0..rows*m) = A p  for the local block of rows [lo, hi).
 *
 *   p_above[0..m)  : the row just above row lo  (may be NULL if lo == 0)
 *   p_local[0..rows*m) : local rows of p  (rows = hi - lo)
 *   p_below[0..m)  : the row just below row hi-1 (may be NULL if hi == m)
 *   lo, hi, m      : row range and grid width
 */
static void apply_A_local(double *q,
                           const double *p_above,
                           const double *p_local,
                           const double *p_below,
                           long lo, long hi, long m)
{
    long rows = hi - lo;
    for (long li = 0; li < rows; li++) {
        long gi = lo + li;          /* global row index */
        for (long j = 0; j < m; j++) {
            long lk = li * m + j;   /* local flat index */
            double val = 4.0 * p_local[lk];

            /* row above */
            if (gi > 0) {
                if (li > 0)
                    val -= p_local[(li - 1) * m + j];
                else
                    val -= p_above[j];          /* halo from neighbour above */
            }
            /* row below */
            if (gi < m - 1) {
                if (li < rows - 1)
                    val -= p_local[(li + 1) * m + j];
                else
                    val -= p_below[j];          /* halo from neighbour below */
            }
            /* left neighbour */
            if (j > 0)  val -= p_local[lk - 1];
            /* right neighbour */
            if (j < m - 1) val -= p_local[lk + 1];

            q[lk] = val;
        }
    }
}

/* ------------------------------------------------------------------ kernel */

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block-partition rows across ranks. */
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long rows = hi - lo;        /* number of rows owned by this rank */
    long n_local = rows * m;    /* number of elements owned locally  */

    /* Allocate local vectors (slices of the full m*m vectors). */
    double *x_loc = (double *)calloc(n_local, sizeof(double)); /* x = 0 */
    double *r_loc = (double *)malloc(n_local * sizeof(double));
    double *p_loc = (double *)malloc(n_local * sizeof(double));
    double *q_loc = (double *)malloc(n_local * sizeof(double));

    /* Halo buffers: one row each from the rank above and below. */
    double *halo_above = (double *)malloc(m * sizeof(double));
    double *halo_below = (double *)malloc(m * sizeof(double));

    /* Initialise: r = b[lo..hi), p = r. */
    memcpy(r_loc, b + lo * m, n_local * sizeof(double));
    memcpy(p_loc, r_loc,      n_local * sizeof(double));

    /* rr = dot(r, r)  -- global */
    double rr_local = local_dot(r_loc, r_loc, n_local);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Neighbour ranks (MPI_PROC_NULL at the grid boundary). */
    int rank_above = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iters; iter++) {

        /* ---- halo exchange: send our top row up, receive from above;
                               send our bottom row down, receive from below. */

        /* Send top row to rank_above, receive bottom halo from rank_above. */
        MPI_Sendrecv(p_loc,                          /* send: first row of p  */
                     (int)m, MPI_DOUBLE, rank_above, 0,
                     halo_below,                     /* recv: row below hi-1  */
                     (int)m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send bottom row to rank_below, receive top halo from rank_below. */
        MPI_Sendrecv(p_loc + (rows - 1) * m,         /* send: last row of p   */
                     (int)m, MPI_DOUBLE, rank_below, 1,
                     halo_above,                      /* recv: row above lo    */
                     (int)m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* ---- q = A p (local block) */
        apply_A_local(q_loc,
                      (rank_above != MPI_PROC_NULL) ? halo_above : NULL,
                      p_loc,
                      (rank_below != MPI_PROC_NULL) ? halo_below : NULL,
                      lo, hi, m);

        /* ---- alpha = rr / dot(p, q) */
        double pq_local = local_dot(p_loc, q_loc, n_local);
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* ---- x += alpha * p;   r -= alpha * q */
        for (long k = 0; k < n_local; k++) {
            x_loc[k] += alpha * p_loc[k];
            r_loc[k] -= alpha * q_loc[k];
        }

        /* ---- rr_new = dot(r, r) */
        double rr_new_local = local_dot(r_loc, r_loc, n_local);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        /* ---- p = r + beta * p */
        for (long k = 0; k < n_local; k++)
            p_loc[k] = r_loc[k] + beta * p_loc[k];
    }

    /* ---- Assemble full x on every rank with MPI_Allgatherv. */

    /* Build counts and displacements (in doubles). */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long rlo = m * r / size;
        long rhi = m * (r + 1) / size;
        recvcounts[r] = (int)((rhi - rlo) * m);
        displs[r]     = (int)(rlo * m);
    }

    MPI_Allgatherv(x_loc, (int)n_local, MPI_DOUBLE,
                   x,     recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(x_loc);
    free(r_loc);
    free(p_loc);
    free(q_loc);
    free(halo_above);
    free(halo_below);
}
