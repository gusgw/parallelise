#include "jacobi.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Jacobi 5-point heat stencil on an n x n row-major grid. For `iters` sweeps, replace
 * each INTERIOR cell by the average of its four neighbours, using the PREVIOUS sweep's
 * values (double-buffered):
 *     u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).
 * The four edges are FIXED boundaries and are never modified.
 *
 * Collective routine: every MPI rank calls it together with an identical initial grid,
 * and on return every rank holds the full final grid u. Decompose the grid rows across
 * the ranks; each sweep, exchange the boundary ("halo") rows with the neighbouring ranks
 * (MPI_Sendrecv) so every rank can update its own rows from the previous sweep; assemble
 * the full grid on every rank at the end. */
void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose interior rows (rows 1..n-2) across ranks */
    long interior_rows = n - 2;
    long lo = interior_rows * rank / size;
    long hi = interior_rows * (rank + 1) / size;
    long my_rows = hi - lo;

    /* Global row indices: rank owns rows (lo+1)..(hi) inclusive (1-based interior) */
    long global_start = lo + 1;

    /* Allocate local buffer: my_rows + 2 halo rows */
    long local_n = my_rows + 2;
    double *local_u = malloc(local_n * n * sizeof(double));
    double *local_un = malloc(local_n * n * sizeof(double));

    /* Copy my rows from u (including halos from neighbors or boundaries) */
    /* local_u[0] is halo from above (row global_start-1) */
    /* local_u[1..my_rows] are my interior rows */
    /* local_u[my_rows+1] is halo from below (row global_start+my_rows) */
    for (long i = 0; i < local_n; i++) {
        long global_i = global_start - 1 + i;
        memcpy(&local_u[i * n], &u[global_i * n], n * sizeof(double));
    }

    /* Determine neighbors */
    int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halos: send my first interior row up, receive into top halo */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_rank, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Exchange halos: send my last interior row down, receive into bottom halo */
        MPI_Sendrecv(&local_u[my_rows * n], n, MPI_DOUBLE, down_rank, 0,
                     &local_u[(my_rows + 1) * n], n, MPI_DOUBLE, down_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Copy to preserve boundaries */
        memcpy(local_un, local_u, local_n * n * sizeof(double));

        /* Update my interior cells */
        for (long i = 1; i <= my_rows; i++) {
            long global_i = global_start - 1 + i;
            /* Skip if this is a global boundary row */
            if (global_i == 0 || global_i == n - 1) continue;

            for (long j = 1; j < n - 1; j++) {
                local_un[i * n + j] = 0.25 * (local_u[(i - 1) * n + j] +
                                               local_u[(i + 1) * n + j] +
                                               local_u[i * n + (j - 1)] +
                                               local_u[i * n + (j + 1)]);
            }
        }

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }

    /* Gather all rows back to u on all ranks */
    /* Each rank sends its interior rows (local_u[1..my_rows]) */
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        long r_lo = interior_rows * r / size;
        long r_hi = interior_rows * (r + 1) / size;
        long r_rows = r_hi - r_lo;
        recvcounts[r] = r_rows * n;
        displs[r] = (r_lo + 1) * n;  /* offset in u (skip top boundary row) */
    }

    MPI_Allgatherv(&local_u[1 * n], my_rows * n, MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(local_u);
    free(local_un);
}
