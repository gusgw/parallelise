/* cg.c — parallel CG solver for the 2-D Poisson system (5-point stencil).
 *
 * Decomposition: rows of the m×m grid are split across MPI ranks.
 *   rank r owns rows [lo, hi)  where
 *     lo = m * r       / size
 *     hi = m * (r + 1) / size
 *
 * Each rank keeps full-length vectors x, r, p, q (length m*m) so that the
 * halo-exchange and Allgatherv bookkeeping is straightforward, but it only
 * WRITES to its own rows [lo*m, hi*m).  The halo exchange copies one ghost
 * row from the rank above and one from the rank below into p before each
 * matrix-vector product.  Two MPI_Allreduce calls per iteration combine the
 * partial dot-products.  One MPI_Allgatherv at the very end assembles the
 * full solution on every rank.
 */

#include "cg.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

/* Partial dot product over the local row range [lo*m, hi*m). */
static double local_dot(const double *u, const double *v, long lo, long hi, long m)
{
    double s = 0.0;
    long start = lo * m;
    long end   = hi * m;
    for (long k = start; k < end; k++) s += u[k] * v[k];
    return s;
}

/* Apply the 5-point Poisson operator to p for rows [lo, hi).
 * Ghost rows p[(lo-1)*m .. lo*m-1]  and  p[hi*m .. (hi+1)*m-1]  must already
 * be filled by the caller (halo exchange).  Rows outside [0, m) are treated
 * as zero (Dirichlet). */
static void apply_A_local(double *q, const double *p,
                           long lo, long hi, long m)
{
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            double s = 4.0 * p[idx];
            if (i > 0)     s -= p[idx - m];
            if (i < m - 1) s -= p[idx + m];
            if (j > 0)     s -= p[idx - 1];
            if (j < m - 1) s -= p[idx + 1];
            q[idx] = s;
        }
    }
}

/* ------------------------------------------------------------------ kernel */

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Row decomposition --------------------------------------------------- */
    long lo = m *  rank      / size;   /* first row owned by this rank */
    long hi = m * (rank + 1) / size;   /* one-past-last row            */
    long local_rows = hi - lo;

    long n = m * m;

    /* Allocate working vectors (full length for simplicity) --------------- */
    double *r = (double *)malloc(sizeof(double) * n);
    double *p = (double *)malloc(sizeof(double) * n);
    double *q = (double *)malloc(sizeof(double) * n);

    /* Allgatherv displacement / count arrays ------------------------------ */
    int *counts = (int *)malloc(sizeof(int) * size);
    int *displs = (int *)malloc(sizeof(int) * size);
    for (int rk = 0; rk < size; rk++) {
        long rlo = m *  rk      / size;
        long rhi = m * (rk + 1) / size;
        counts[rk] = (int)((rhi - rlo) * m);
        displs[rk] = (int)(rlo * m);
    }

    /* Neighbour ranks (MPI_PROC_NULL at boundaries) ----------------------- */
    int rank_above = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Initialise: x = 0, r = b, p = r  (only local rows matter) ---------- */
    memset(x, 0, sizeof(double) * n);
    memcpy(r, b, sizeof(double) * n);
    memcpy(p, b, sizeof(double) * n);

    /* rr = dot(r, r)  — global, computed from local partial sums ---------- */
    double local_rr = local_dot(r, r, lo, hi, m);
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* CG iterations ------------------------------------------------------- */
    for (int iter = 0; iter < iters; iter++) {

        /* Halo exchange: send our top row up, receive ghost from above;
         *                send our bottom row down, receive ghost from below. */
        double *send_up   = p + lo * m;            /* our first row → rank above */
        double *recv_up   = p + (lo - 1) * m;      /* ghost row above us         */
        double *send_down = p + (hi - 1) * m;      /* our last row  → rank below */
        double *recv_down = p + hi * m;             /* ghost row below us         */

        /* Guard: recv_up / recv_down may point outside [0,n) when rank_above /
         * rank_below is MPI_PROC_NULL — MPI_Sendrecv will not touch them in
         * that case, so we just need a valid (non-NULL) buffer pointer.
         * Re-use send buffers as dummy receive buffers for the NULL direction. */
        if (rank_above == MPI_PROC_NULL) recv_up   = send_up;
        if (rank_below == MPI_PROC_NULL) recv_down = send_down;

        /* Send top row to rank above, receive ghost from rank above */
        MPI_Sendrecv(send_up,   (int)m, MPI_DOUBLE, rank_above, 0,
                     recv_up,   (int)m, MPI_DOUBLE, rank_above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send bottom row to rank below, receive ghost from rank below */
        MPI_Sendrecv(send_down, (int)m, MPI_DOUBLE, rank_below, 1,
                     recv_down, (int)m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* q = A p  (local rows only) */
        apply_A_local(q, p, lo, hi, m);

        /* alpha = rr / dot(p, q) */
        double local_pq = local_dot(p, q, lo, hi, m);
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* x += alpha * p;  r -= alpha * q  (local rows) */
        for (long k = lo * m; k < hi * m; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }

        /* rr_new = dot(r, r) */
        double local_rr_new = local_dot(r, r, lo, hi, m);
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p  (local rows) */
        for (long k = lo * m; k < hi * m; k++) {
            p[k] = r[k] + beta * p[k];
        }
    }

    /* Assemble full x on every rank --------------------------------------- */
    /* Each rank contributes its own slice; Allgatherv writes directly into x. */
    MPI_Allgatherv(MPI_IN_PLACE, (int)(local_rows * m), MPI_DOUBLE,
                   x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(r);
    free(p);
    free(q);
    free(counts);
    free(displs);
}
