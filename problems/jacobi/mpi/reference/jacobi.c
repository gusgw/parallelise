#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Full grid is
 * replicated on every rank; rank owns rows [lo, hi). Each sweep: compute the new values
 * for the owned rows from the previous grid, copy them back into u, then exchange only
 * the two boundary rows with the up/down neighbours (MPI_Sendrecv). A single final
 * MPI_Allgatherv assembles the whole grid on every rank. Collective; MPI already init'd. */
void jacobi(double *u, long n, int iters)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;
    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double *unew = malloc(sizeof(double) * n * n);

    for (int it = 0; it < iters; it++) {
        for (long i = lo; i < hi; i++)
            for (long j = 0; j < n; j++)
                unew[i * n + j] = (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                    ? u[i * n + j]      /* fixed boundary */
                    : 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j]
                              + u[i * n + j - 1] + u[i * n + j + 1]);
        memcpy(&u[lo * n], &unew[lo * n], sizeof(double) * (size_t)(hi - lo) * n);

        /* exchange the freshly computed boundary rows with the neighbours (single tag:
         * messages are disambiguated by source rank). Edge halos use a valid dummy
         * pointer since the PROC_NULL send/recv is a no-op. */
        double *top_halo = (lo > 0) ? &u[(lo - 1) * n] : u;
        double *bot_halo = (hi < n) ? &u[hi * n] : u;
        MPI_Sendrecv(&u[lo * n], (int)n, MPI_DOUBLE, up, 0,
                     top_halo, (int)n, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[(hi - 1) * n], (int)n, MPI_DOUBLE, down, 0,
                     bot_halo, (int)n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long rl = n * r / size, rh = n * (r + 1) / size;
        counts[r] = (int)((rh - rl) * n);
        displs[r] = (int)(rl * n);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   u, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    free(counts);
    free(displs);
    free(unew);
}
