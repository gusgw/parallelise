#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include "nbody.h"

/* Reference solution (used only to validate the harness -- NOT shown to the agents).
 * Block decomposition of the bodies: each rank computes the accelerations for its
 * bodies [lo, hi) (the positions are read-only and every rank already holds them all,
 * so no halo exchange is needed), then MPI_Allgatherv assembles the full ax and ay on
 * every rank. Collective; assumes MPI already initialised. */
void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double eps2 = 0.0025;   /* softening 0.05 squared */
    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++) {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0, sy = 0.0;
        for (long j = 0; j < n; j++) {
            if (j == i)
                continue;
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double r2 = dx * dx + dy * dy;
            const double d3 = (r2 + eps2) * sqrt(r2 + eps2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }

    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long rlo = n * r / size, rhi = n * (r + 1) / size;
        counts[r] = (int)(rhi - rlo);
        displs[r] = (int)rlo;
    }
    /* each rank's block is already at displs[rank] in ax and ay -> gather in place */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   ax, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   ay, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    free(counts);
    free(displs);
}
