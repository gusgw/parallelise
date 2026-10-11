/* cg.c — parallel CG solver for the 2-D Poisson system (MPI, row decomposition).
 *
 * Each rank owns a contiguous band of rows [lo, hi) of the m x m grid.
 * Before every matvec the boundary rows of p are exchanged with neighbours
 * (MPI_Sendrecv; MPI_PROC_NULL at the domain edges avoids special-casing).
 * Dot products are reduced with MPI_Allreduce so every rank has alpha/beta.
 * The full solution is assembled on every rank with MPI_Allgatherv.
 */

#include "cg.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

/* Local dot product over a flat array of length n. */
static double local_dot(const double *u, const double *v, long n)
{
    double s = 0.0;
    for (long k = 0; k < n; k++) s += u[k] * v[k];
    return s;
}

/* Global dot product: reduce local partial sums across all ranks. */
static double global_dot(const double *u, const double *v, long n,
                         MPI_Comm comm)
{
    double local = local_dot(u, v, n);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

/* Apply the 5-point Poisson operator to the local rows [lo, hi) of p.
 *
 * p_local[0 .. local_rows-1][0 .. m-1]  — the owned rows
 * halo_above[0 .. m-1]                  — row lo-1 (or zeros if lo==0)
 * halo_below[0 .. m-1]                  — row hi   (or zeros if hi==m)
 *
 * Result written into q_local (same shape as p_local).
 */
static void apply_A_local(const double *p_local,
                           const double *halo_above,
                           const double *halo_below,
                           double       *q_local,
                           long lo, long hi, long m)
{
    long local_rows = hi - lo;
    for (long li = 0; li < local_rows; li++) {
        long gi = lo + li;                  /* global row index */
        for (long j = 0; j < m; j++) {
            long lk = li * m + j;           /* local flat index */
            double s = 4.0 * p_local[lk];

            /* row above */
            if (li > 0)
                s -= p_local[(li - 1) * m + j];
            else if (gi > 0)
                s -= halo_above[j];
            /* (else gi==0: Dirichlet, contributes 0) */

            /* row below */
            if (li < local_rows - 1)
                s -= p_local[(li + 1) * m + j];
            else if (gi < m - 1)
                s -= halo_below[j];
            /* (else gi==m-1: Dirichlet, contributes 0) */

            /* left neighbour */
            if (j > 0)  s -= p_local[lk - 1];

            /* right neighbour */
            if (j < m - 1) s -= p_local[lk + 1];

            q_local[lk] = s;
        }
    }
}

/* ------------------------------------------------------------------ kernel */

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Row decomposition: rank r owns global rows [lo, hi). */
    long lo = (long)rank       * m / size;
    long hi = (long)(rank + 1) * m / size;
    long local_rows = hi - lo;
    long local_n    = local_rows * m;   /* number of owned unknowns */

    /* Allocate local working arrays. */
    double *xl = (double *)calloc((size_t)local_n, sizeof(double)); /* x local, init 0 */
    double *rl = (double *)malloc((size_t)local_n * sizeof(double));
    double *pl = (double *)malloc((size_t)local_n * sizeof(double));
    double *ql = (double *)malloc((size_t)local_n * sizeof(double));
    double *halo_above = (double *)calloc((size_t)m, sizeof(double));
    double *halo_below = (double *)calloc((size_t)m, sizeof(double));

    /* r = b (local slice); p = r */
    memcpy(rl, b + lo * m, (size_t)local_n * sizeof(double));
    memcpy(pl, rl,         (size_t)local_n * sizeof(double));

    double rr = global_dot(rl, rl, local_n, MPI_COMM_WORLD);

    /* Neighbour ranks (MPI_PROC_NULL at domain boundaries). */
    int rank_above = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iters; iter++) {

        /* ---- halo exchange: send/recv one row of p ---- */

        /* Send our top row up, receive from below into halo_below. */
        MPI_Sendrecv(pl,                        /* send top row of local p   */
                     (int)m, MPI_DOUBLE, rank_above, 0,
                     halo_below,                /* recv row hi from below    */
                     (int)m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send our bottom row down, receive from above into halo_above. */
        MPI_Sendrecv(pl + (local_rows - 1) * m, /* send bottom row of local p */
                     (int)m, MPI_DOUBLE, rank_below, 1,
                     halo_above,                /* recv row lo-1 from above  */
                     (int)m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* ---- q = A p (local) ---- */
        apply_A_local(pl, halo_above, halo_below, ql, lo, hi, m);

        /* ---- alpha = rr / dot(p, q) ---- */
        double pq  = global_dot(pl, ql, local_n, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* ---- x += alpha * p;  r -= alpha * q ---- */
        for (long k = 0; k < local_n; k++) xl[k] += alpha * pl[k];
        for (long k = 0; k < local_n; k++) rl[k] -= alpha * ql[k];

        /* ---- beta = rr_new / rr ---- */
        double rr_new = global_dot(rl, rl, local_n, MPI_COMM_WORLD);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* ---- p = r + beta * p ---- */
        for (long k = 0; k < local_n; k++) pl[k] = rl[k] + beta * pl[k];
    }

    /* ---- Assemble full x on every rank via MPI_Allgatherv ---- */

    /* Build recvcounts and displacements (in doubles). */
    int *recvcounts = (int *)malloc((size_t)size * sizeof(int));
    int *displs     = (int *)malloc((size_t)size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long r_lo = (long)r       * m / size;
        long r_hi = (long)(r + 1) * m / size;
        recvcounts[r] = (int)((r_hi - r_lo) * m);
        displs[r]     = (int)(r_lo * m);
    }

    MPI_Allgatherv(xl, (int)local_n, MPI_DOUBLE,
                   x,  recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(xl); free(rl); free(pl); free(ql);
    free(halo_above); free(halo_below);
    free(recvcounts); free(displs);
}
