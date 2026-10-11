/* jacobi.c — parallel 2-D Jacobi heat stencil (MPI, row decomposition).
 *
 * Each rank owns a contiguous band of rows [lo, hi) of the n×n grid.
 * Every sweep:
 *   1. Exchange one halo row with the rank above (rank-1) and below (rank+1)
 *      using MPI_Sendrecv with MPI_PROC_NULL for edge ranks that have no
 *      neighbour.
 *   2. Update every interior cell in the local band from the previous values
 *      (double-buffered; boundary rows i==0 and i==n-1 are never modified).
 * After all sweeps, MPI_Allgatherv reassembles the full grid on every rank.
 */

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

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
    long nlocal = hi - lo;                   /* number of rows this rank   */

    /* Neighbour ranks (MPI_PROC_NULL for boundary ranks). */
    int rank_above = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* ------------------------------------------------------------------ *
     * Local working buffers: nlocal rows + 2 halo rows (top & bottom).   *
     * Layout: halo_top | row[lo] … row[hi-1] | halo_bot                  *
     * We keep two such buffers and ping-pong between them.                *
     * ------------------------------------------------------------------ */
    long buf_rows = nlocal + 2;          /* includes the two halo rows     */
    long buf_size = buf_rows * n;

    double *a = (double *)malloc(sizeof(double) * buf_size);
    double *b = (double *)malloc(sizeof(double) * buf_size);
    if (!a || !b) { MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Copy this rank's rows from the caller-supplied grid into a[].
     * a[0*n .. 1*n-1]          = halo_top  (filled by MPI_Sendrecv)
     * a[1*n .. (nlocal)*n-1]   = rows lo .. hi-1
     * a[(nlocal+1)*n .. end-1] = halo_bot  (filled by MPI_Sendrecv)    */
    memcpy(a + n,                        /* dest: first real row in buf   */
           u + lo * n,                   /* src:  row lo in global grid   */
           sizeof(double) * nlocal * n);

    /* ------------------------------------------------------------------ *
     * Sweep loop.                                                          *
     * ------------------------------------------------------------------ */
    for (int iter = 0; iter < iters; iter++) {

        /* -- halo exchange -------------------------------------------- */
        /* Send our top real row upward; receive into our top halo.       */
        MPI_Sendrecv(
            a + 1 * n,          /* send: our first real row               */
            (int)n, MPI_DOUBLE, rank_above, 0,
            a + 0 * n,          /* recv: halo_top                         */
            (int)n, MPI_DOUBLE, rank_above, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send our bottom real row downward; receive into our bot halo.  */
        MPI_Sendrecv(
            a + nlocal * n,     /* send: our last real row                */
            (int)n, MPI_DOUBLE, rank_below, 0,
            a + (nlocal + 1) * n, /* recv: halo_bot                       */
            (int)n, MPI_DOUBLE, rank_below, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* -- copy a → b (preserves fixed boundary values) ------------- */
        memcpy(b, a, sizeof(double) * buf_size);

        /* -- stencil update ------------------------------------------- *
         * Global row index of the cell in a[r*n+j] is  lo + (r-1).      *
         * We must not touch global rows 0 or n-1 (fixed boundaries).     *
         * In local index space that means:                                *
         *   r=1 corresponds to global row lo   → skip if lo==0           *
         *   r=nlocal corresponds to global row hi-1 → skip if hi-1==n-1  */
        long r_start = 1;                        /* local buf row index    */
        long r_end   = nlocal;                   /* inclusive              */
        if (lo == 0)     r_start = 2;            /* skip global row 0      */
        if (hi == n)     r_end   = nlocal - 1;   /* skip global row n-1    */

        for (long r = r_start; r <= r_end; r++) {
            for (long j = 1; j < n - 1; j++) {
                b[r * n + j] = 0.25 * (a[(r - 1) * n + j] +
                                       a[(r + 1) * n + j] +
                                       a[r * n + j - 1]   +
                                       a[r * n + j + 1]);
            }
        }

        /* -- ping-pong ------------------------------------------------ */
        double *tmp = a; a = b; b = tmp;
    }

    /* ------------------------------------------------------------------ *
     * Assemble the full grid on every rank via MPI_Allgatherv.            *
     * Each rank contributes its nlocal rows (without the halo rows).      *
     * ------------------------------------------------------------------ */
    int *recvcounts = (int *)malloc(sizeof(int) * size);
    int *displs     = (int *)malloc(sizeof(int) * size);
    if (!recvcounts || !displs) { MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int r = 0; r < size; r++) {
        long rlo = (long)r       * n / size;
        long rhi = (long)(r + 1) * n / size;
        recvcounts[r] = (int)((rhi - rlo) * n);
        displs[r]     = (int)(rlo * n);
    }

    /* Source: a + n  (skip the halo_top row; send nlocal real rows).     */
    MPI_Allgatherv(
        a + n,                  /* sendbuf: our real rows                 */
        (int)(nlocal * n),      /* sendcount                              */
        MPI_DOUBLE,
        u,                      /* recvbuf: caller's grid                 */
        recvcounts,
        displs,
        MPI_DOUBLE,
        MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(a);
    free(b);
}
