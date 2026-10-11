/* nbody.c — Direct 2-D N-body acceleration kernel, parallelised with OpenMP.
 *
 * For every body i we compute the direct sum over all j != i:
 *   ax[i] = sum_{j!=i}  (px[j]-px[i]) / d3
 *   ay[i] = sum_{j!=i}  (py[j]-py[i]) / d3
 * where  d3 = (r2 + EPS2) * sqrt(r2 + EPS2),  r2 = dx*dx + dy*dy,
 * and EPS2 = 0.0025 (softening = 0.05 squared).
 *
 * Each body i is independent of every other body i', so the outer loop is
 * trivially parallel.  The inner loop accumulates into private scalars sx/sy
 * and writes to ax[i]/ay[i] only once per i, so there are no data races.
 */

#include "nbody.h"

#include <math.h>   /* sqrt */

#define EPS2 0.0025

void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        double xi = px[i];
        double yi = py[i];
        double sx = 0.0;
        double sy = 0.0;

        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - xi;
            double dy = py[j] - yi;
            double r2 = dx * dx + dy * dy;
            double r2s = r2 + EPS2;
            double d3  = r2s * sqrt(r2s);
            sx += dx / d3;
            sy += dy / d3;
        }

        ax[i] = sx;
        ay[i] = sy;
    }
}
