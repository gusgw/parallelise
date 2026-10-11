#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "kernel_b.h"
#include "problem.h"

void kernel_b(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_rows = hi - lo;

    double *cur = (double *)malloc((local_rows + 2) * n * sizeof(double));
    double *nxt = (double *)malloc((local_rows + 2) * n * sizeof(double));
    if (!cur || !nxt) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialise owned rows (skip halo rows) */
    for (long i = 0; i < local_rows; ++i) {
        long gi = lo + i;
        for (long j = 0; j < n; ++j) {
            cur[(i + 1) * n + j] = u_init(gi, j, n);
        }
    }
    /* Halo rows will be overwritten by exchanges */
    memset(cur, 0, n * sizeof(double));                     /* top halo */
    memset(cur + (local_rows + 1) * n, 0, n * sizeof(double)); /* bottom halo */

    for (int it = 0; it < iters; ++it) {
        int up   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        /* Exchange top halo: send first owned row up, receive from up into top halo */
        MPI_Sendrecv(cur + 1 * n, n, MPI_DOUBLE, up, 0,
                     cur + 0 * n, n, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Exchange bottom halo: send last owned row down, receive from down into bottom halo */
        MPI_Sendrecv(cur + local_rows * n, n, MPI_DOUBLE, down, 0,
                     cur + (local_rows + 1) * n, n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (long i = 1; i <= local_rows; ++i) {
            long gi = lo + i - 1;
            if (gi == 0 || gi == n - 1) {
                memcpy(&nxt[i * n], &cur[i * n], n * sizeof(double));
                continue;
            }
            nxt[i * n + 0] = cur[i * n + 0];
            nxt[i * n + (n - 1)] = cur[i * n + (n - 1)];
            for (long j = 1; j < n - 1; ++j) {
                nxt[i * n + j] = 0.25 * (cur[(i - 1) * n + j] +
                                         cur[(i + 1) * n + j] +
                                         cur[i * n + (j - 1)] +
                                         cur[i * n + (j + 1)]);
            }
        }
        double *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    /* Gather the full grid on all ranks */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        long rlo = n * r / size;
        long rhi = n * (r + 1) / size;
        recvcounts[r] = (int)((rhi - rlo) * n);
        displs[r] = (int)(rlo * n);
    }
    double *owned = cur + n; /* first owned row after top halo */
    MPI_Allgatherv(owned, (int)(local_rows * n), MPI_DOUBLE,
                   u, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(cur);
    free(nxt);
}
