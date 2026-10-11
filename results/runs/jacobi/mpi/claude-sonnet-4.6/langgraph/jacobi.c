/* jacobi.c — parallel 2-D Jacobi heat stencil (MPI, row decomposition).
 *
 * Each rank owns a contiguous band of rows [lo, hi) of the n×n grid.
 * Every sweep:
 *   1. Exchange the top/bottom halo rows with the neighbouring ranks via
 *      MPI_Sendrecv (MPI_PROC_NULL guards the edge ranks).
 *   2. Update only the interior cells that belong to this rank.
 * After all sweeps, MPI_Allgatherv reassembles the full grid on every rank.
 */

#include "jacobi.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ------------------------------------------------------------------ *
     * Row decomposition: rank r owns rows [lo, hi).                       *
     * ------------------------------------------------------------------ */
    long lo = (long)rank       * n / size;   /* first row owned            */
    long hi = (long)(rank + 1) * n / size;   /* one-past-last row owned    */
    long nlocal = hi - lo;                   /* number of rows on this rank */

    /* Neighbours in the ring of ranks (MPI_PROC_NULL for edge ranks). */
    int rank_above = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* ------------------------------------------------------------------ *
     * Local buffer: nlocal rows + 1 halo row above + 1 halo row below.   *
     * Layout: row 0 = halo-above, rows 1..nlocal = owned, row nlocal+1   *
     * = halo-below.                                                       *
     * ------------------------------------------------------------------ */
    long buf_rows = nlocal + 2;          /* includes both halo rows        */
    double *cur  = (double *)malloc(sizeof(double) * buf_rows * n);
    double *next = (double *)malloc(sizeof(double) * buf_rows * n);

    /* Copy the owned rows from the caller's grid into cur (offset by 1   *
     * for the halo row above).                                            */
    memcpy(cur + n, u + lo * n, sizeof(double) * nlocal * n);

    /* Initialise halo rows to 0 (they will be filled by Sendrecv). */
    memset(cur,              0, sizeof(double) * n);          /* halo above */
    memset(cur + (nlocal + 1) * n, 0, sizeof(double) * n);   /* halo below */

    /* ------------------------------------------------------------------ *
     * Sweep loop.                                                         *
     * ------------------------------------------------------------------ */
    for (int iter = 0; iter < iters; iter++) {

        /* -- Halo exchange -------------------------------------------- */
        /* Send our first owned row up, receive the halo from above.      */
        MPI_Sendrecv(
            cur + n,              n, MPI_DOUBLE, rank_above, 0,  /* send */
            cur,                  n, MPI_DOUBLE, rank_above, 1,  /* recv */
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send our last owned row down, receive the halo from below.     */
        MPI_Sendrecv(
            cur + nlocal * n,     n, MPI_DOUBLE, rank_below, 1,  /* send */
            cur + (nlocal+1) * n, n, MPI_DOUBLE, rank_below, 0,  /* recv */
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* -- Copy cur → next (preserves fixed boundary rows). ---------- */
        memcpy(next, cur, sizeof(double) * buf_rows * n);

        /* -- Stencil update for interior cells owned by this rank. ----- *
         * Global row 0 (top edge) and row n-1 (bottom edge) are fixed.   *
         * Global column 0 and column n-1 (left/right edges) are fixed.   *
         * In local indexing, global row g maps to local index g - lo + 1. */
        for (long li = 1; li <= nlocal; li++) {
            long gi = lo + li - 1;          /* global row index            */
            if (gi == 0 || gi == n - 1) continue;   /* fixed boundary row */

            for (long j = 1; j < n - 1; j++) {
                next[li * n + j] = 0.25 * (
                    cur[(li - 1) * n + j] +   /* row above (may be halo)  */
                    cur[(li + 1) * n + j] +   /* row below (may be halo)  */
                    cur[li * n + j - 1]     + /* left neighbour           */
                    cur[li * n + j + 1]);     /* right neighbour          */
            }
        }

        /* Swap buffers. */
        double *tmp = cur;
        cur  = next;
        next = tmp;
    }

    /* ------------------------------------------------------------------ *
     * Reassemble the full grid on every rank via MPI_Allgatherv.         *
     * Each rank contributes its nlocal rows (no halo rows).              *
     * ------------------------------------------------------------------ */

    /* Build the displacement and count arrays. */
    int *counts = (int *)malloc(sizeof(int) * size);
    int *displs = (int *)malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long r_lo = (long)r       * n / size;
        long r_hi = (long)(r + 1) * n / size;
        counts[r] = (int)((r_hi - r_lo) * n);
        displs[r] = (int)(r_lo * n);
    }

    /* Gather into the caller's buffer u (every rank gets the full grid). */
    MPI_Allgatherv(
        cur + n,          /* send: owned rows (skip halo-above)            */
        (int)(nlocal * n),
        MPI_DOUBLE,
        u,                /* recv: full n×n grid                           */
        counts, displs,
        MPI_DOUBLE,
        MPI_COMM_WORLD);

    free(counts);
    free(displs);
    free(cur);
    free(next);
}
