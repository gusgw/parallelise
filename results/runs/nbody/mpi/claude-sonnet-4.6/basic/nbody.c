/* nbody.c — parallel 2-D direct N-body acceleration kernel (MPI).
 *
 * Each rank computes the acceleration for its own block of bodies i in
 * [lo, hi), then MPI_Allgatherv assembles the full ax/ay on every rank.
 * The positions px/py are read-only and already replicated on every rank,
 * so no halo exchange is needed.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "nbody.h"

#define EPS2 0.0025   /* softening: 0.05^2 */

void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Block decomposition: rank r owns bodies [lo, hi) */
    long lo = (long)rank       * n / (long)size;
    long hi = (long)(rank + 1) * n / (long)size;
    long local_n = hi - lo;

    /* Allocate local buffers for this rank's slice */
    double *lax = (double *)malloc(sizeof(double) * (local_n > 0 ? local_n : 1));
    double *lay = (double *)malloc(sizeof(double) * (local_n > 0 ? local_n : 1));
    if (!lax || !lay) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Compute accelerations for the local block of bodies */
    for (long i = lo; i < hi; i++) {
        double sx = 0.0, sy = 0.0;
        double pxi = px[i], pyi = py[i];
        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - pxi;
            double dy = py[j] - pyi;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        lax[i - lo] = sx;
        lay[i - lo] = sy;
    }

    /* Build the per-rank counts and displacements for Allgatherv */
    int *counts = (int *)malloc(sizeof(int) * (size_t)size);
    int *displs = (int *)malloc(sizeof(int) * (size_t)size);
    if (!counts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int r = 0; r < size; r++) {
        long r_lo = (long)r       * n / (long)size;
        long r_hi = (long)(r + 1) * n / (long)size;
        counts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }

    /* Assemble the full ax and ay on every rank */
    MPI_Allgatherv(lax, (int)local_n, MPI_DOUBLE,
                   ax, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(lay, (int)local_n, MPI_DOUBLE,
                   ay, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(lax);
    free(lay);
    free(counts);
    free(displs);
}
