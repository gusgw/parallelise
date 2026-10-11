/*
 * jacobi.c — parallel 2-D Jacobi heat stencil (MPI, row decomposition).
 *
 * The n×n grid is decomposed in the row dimension across `size` ranks:
 *   rank r owns rows  [lo, hi)  where
 *     lo = n * r       / size
 *     hi = n * (r + 1) / size
 *
 * Each sweep:
 *   1. Exchange halo rows with the rank above (rank-1) and below (rank+1)
 *      using MPI_Sendrecv so that every rank can compute the full 5-point
 *      stencil for its interior rows.
 *   2. Update every interior cell in [lo, hi) that is not on a global edge.
 *
 * After all sweeps, MPI_Allgatherv reassembles the full grid on every rank.
 *
 * The caller (main) owns the full n×n array `u` (row-major, initialised with
 * boundary conditions).  On return every rank holds the identical final grid.
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

    /* ------------------------------------------------------------------ */
    /* 1.  Row decomposition: rank r owns global rows [lo, hi).            */
    /* ------------------------------------------------------------------ */
    long lo = n *  rank      / size;   /* first row owned by this rank     */
    long hi = n * (rank + 1) / size;   /* one-past-last row owned          */
    long nlocal = hi - lo;             /* number of rows owned             */

    /* Neighbours in the ring (MPI_PROC_NULL for edge ranks). */
    int rank_above = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int rank_below = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    /* ------------------------------------------------------------------ */
    /* 2.  Allocate local working buffers (with one halo row on each side).*/
    /*     Layout: row 0 = halo-above, rows 1..nlocal = owned, row nlocal+1*/
    /*     = halo-below.  Total: (nlocal + 2) rows.                        */
    /* ------------------------------------------------------------------ */
    long stride  = n;                          /* columns per row          */
    long buf_rows = nlocal + 2;                /* +2 halo rows             */

    double *cur  = (double *)malloc(sizeof(double) * buf_rows * stride);
    double *next = (double *)malloc(sizeof(double) * buf_rows * stride);

    /* Copy owned rows from the caller's full grid into cur[1..nlocal]. */
    memcpy(cur  + 1 * stride,
           u    + lo * stride,
           sizeof(double) * nlocal * stride);

    /* Initialise halo rows to 0 (they will be filled by MPI_Sendrecv). */
    memset(cur,                          0, sizeof(double) * stride);
    memset(cur + (nlocal + 1) * stride,  0, sizeof(double) * stride);

    /* ------------------------------------------------------------------ */
    /* 3.  Jacobi sweeps.                                                  */
    /* ------------------------------------------------------------------ */
    for (int iter = 0; iter < iters; iter++) {

        /* --- halo exchange -------------------------------------------- */
        /* Send our first owned row up, receive the halo from above.       */
        MPI_Sendrecv(cur + 1       * stride, (int)n, MPI_DOUBLE, rank_above, 0,
                     cur + 0       * stride, (int)n, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send our last owned row down, receive the halo from below.      */
        MPI_Sendrecv(cur + nlocal  * stride, (int)n, MPI_DOUBLE, rank_below, 0,
                     cur + (nlocal + 1) * stride, (int)n, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* --- copy cur → next (preserves fixed boundary values) --------- */
        memcpy(next, cur, sizeof(double) * buf_rows * stride);

        /* --- stencil update -------------------------------------------- */
        for (long li = 1; li <= nlocal; li++) {
            long gi = lo + (li - 1);   /* global row index                 */

            /* Skip global boundary rows (top row 0, bottom row n-1). */
            if (gi == 0 || gi == n - 1) continue;

            for (long j = 1; j < n - 1; j++) {
                /* Neighbours: li-1 and li+1 in the local buffer (which
                 * includes halo rows), j-1 and j+1 in the same row.      */
                next[li * stride + j] =
                    0.25 * (cur[(li - 1) * stride + j] +
                            cur[(li + 1) * stride + j] +
                            cur[ li      * stride + j - 1] +
                            cur[ li      * stride + j + 1]);
            }
        }

        /* Swap buffers. */
        double *tmp = cur;
        cur  = next;
        next = tmp;
    }

    /* ------------------------------------------------------------------ */
    /* 4.  Gather the full grid on every rank (MPI_Allgatherv).            */
    /* ------------------------------------------------------------------ */

    /* Each rank contributes nlocal rows; counts and displacements in      */
    /* units of doubles.                                                   */
    int *recvcounts = (int *)malloc(sizeof(int) * size);
    int *displs     = (int *)malloc(sizeof(int) * size);

    for (int r = 0; r < size; r++) {
        long r_lo   = n *  r      / size;
        long r_hi   = n * (r + 1) / size;
        recvcounts[r] = (int)((r_hi - r_lo) * n);
        displs[r]     = (int)(r_lo * n);
    }

    /* cur[1..nlocal] holds the final owned rows (local indices 1-based). */
    MPI_Allgatherv(cur + 1 * stride,          /* send: owned rows          */
                   (int)(nlocal * n),
                   MPI_DOUBLE,
                   u,                          /* recv: full grid           */
                   recvcounts,
                   displs,
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(cur);
    free(next);
}
