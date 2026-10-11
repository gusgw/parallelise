/* nbody.c — parallel 2-D direct N-body acceleration kernel (MPI).
 *
 * Each rank computes the acceleration for its own contiguous block of bodies
 * [lo, hi) by summing the contribution of all n bodies (read-only positions are
 * already replicated on every rank).  MPI_Allgatherv then assembles the full
 * ax[] / ay[] arrays on every rank so the caller sees a complete result.
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "nbody.h"

#define EPS2 0.0025   /* softening: 0.05^2 */

void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- domain decomposition: rank r owns bodies [lo, hi) ---- */
    long lo = n * rank       / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* ---- local acceleration buffers ---- */
    double *lax = (double *)malloc(sizeof(double) * (local_n > 0 ? local_n : 1));
    double *lay = (double *)malloc(sizeof(double) * (local_n > 0 ? local_n : 1));
    if (!lax || !lay) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- O(n^2) direct sum for this rank's slice ---- */
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

    /* ---- gather counts and displacements for Allgatherv ---- */
    int *counts = (int *)malloc(sizeof(int) * size);
    int *displs = (int *)malloc(sizeof(int) * size);
    if (!counts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int r = 0; r < size; r++) {
        long r_lo = n * r       / size;
        long r_hi = n * (r + 1) / size;
        counts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }

    /* ---- assemble full ax[] and ay[] on every rank ---- */
    MPI_Allgatherv(lax, (int)local_n, MPI_DOUBLE,
                   ax,  counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(lay, (int)local_n, MPI_DOUBLE,
                   ay,  counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(lax);
    free(lay);
    free(counts);
    free(displs);
}
