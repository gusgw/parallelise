/* nbody.c — parallel MPI kernel for direct 2-D N-body acceleration.
 *
 * Each rank computes the accelerations for its own contiguous block of bodies
 * [lo, hi).  Because every rank already holds the full position arrays (px, py),
 * no halo exchange is needed: the inner j-loop simply walks all n bodies.
 * After the local computation the partial ax/ay slices are assembled on every
 * rank with MPI_Allgatherv so that on return every rank holds the complete
 * ax[0..n-1] and ay[0..n-1].
 *
 * MPI is assumed to be already initialised by the caller; this routine must NOT
 * call MPI_Init or MPI_Finalize.
 */

#include "nbody.h"

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

    /* ---- O(n^2) direct sum for the local block of bodies ---- */
    for (long ii = 0; ii < local_n; ii++) {
        long i = lo + ii;
        double xi = px[i];
        double yi = py[i];
        double sx = 0.0;
        double sy = 0.0;
        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - xi;
            double dy = py[j] - yi;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        lax[ii] = sx;
        lay[ii] = sy;
    }

    /* ---- gather all slices onto every rank ---- */
    /* Build recvcounts and displacements for Allgatherv */
    int *recvcounts = (int *)malloc(sizeof(int) * size);
    int *displs     = (int *)malloc(sizeof(int) * size);
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int r = 0; r < size; r++) {
        long rlo = n * r       / size;
        long rhi = n * (r + 1) / size;
        recvcounts[r] = (int)(rhi - rlo);
        displs[r]     = (int)rlo;
    }

    MPI_Allgatherv(lax, (int)local_n, MPI_DOUBLE,
                   ax,  recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(lay, (int)local_n, MPI_DOUBLE,
                   ay,  recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(lax);
    free(lay);
    free(recvcounts);
    free(displs);
}
