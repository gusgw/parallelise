/*
 * Parallel Jacobi 5‑point stencil implementation.
 * Decomposes the grid rows across MPI ranks, exchanges halo rows each sweep,
 * and leaves the full final grid in the caller's array `u` on every rank.
 * This avoids a costly final gather and yields better parallel speedup.
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

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Allocate two buffers with halo rows on each side. */
    double *cur = malloc((local_n + 2) * n * sizeof(double));
    double *next = malloc((local_n + 2) * n * sizeof(double));
    if (!cur || !next) MPI_Abort(MPI_COMM_WORLD, 1);

    /* Copy owned interior rows into the interior of cur. */
    memcpy(&cur[n], &u[lo * n], local_n * n * sizeof(double));
    /* Initialise halo rows for the global fixed boundaries. */
    if (rank == 0) {
        memcpy(&cur[0], &u[0], n * sizeof(double)); // top boundary
    }
    if (rank == size - 1) {
        memcpy(&cur[(local_n + 1) * n], &u[(n - 1) * n], n * sizeof(double)); // bottom
    }

    MPI_Datatype rowtype;
    MPI_Type_contiguous((int)n, MPI_DOUBLE, &rowtype);
    MPI_Type_commit(&rowtype);

    for (int it = 0; it < iters; ++it) {
        int src = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int dst = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        /* Exchange halo rows with complementary tags. */
        MPI_Sendrecv(&cur[1 * n], 1, rowtype, src, 0,
                     &cur[0 * n], 1, rowtype, src, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[local_n * n], 1, rowtype, dst, 1,
                     &cur[(local_n + 1) * n], 1, rowtype, dst, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Jacobi update on interior rows (skip global fixed rows). */
        for (long i = 1; i <= local_n; ++i) {
            long gi = lo + i - 1;               /* global row index */
            if (gi == 0 || gi == n - 1) {
                memcpy(&next[i * n], &cur[i * n], n * sizeof(double));
                continue;
            }
            for (long j = 1; j < n - 1; ++j) {
                next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] +
                                          cur[(i + 1) * n + j] +
                                          cur[i * n + (j - 1)] +
                                          cur[i * n + (j + 1)]);
            }
            next[i * n + 0] = cur[i * n + 0];
            next[i * n + (n - 1)] = cur[i * n + (n - 1)];
        }

        /* Swap buffers for next iteration. */
        double *tmp = cur;
        cur = next;
        next = tmp;

        /* Restore fixed boundary halo rows after swap. */
        if (rank == 0) memcpy(&cur[0], &u[0], n * sizeof(double));
        if (rank == size - 1) memcpy(&cur[(local_n + 1) * n], &u[(n - 1) * n], n * sizeof(double));

        /* Copy updated interior rows back into the global array `u`. */
        memcpy(&u[lo * n], &cur[n], local_n * n * sizeof(double));
    }

    free(cur);
    free(next);
    MPI_Type_free(&rowtype);
}
