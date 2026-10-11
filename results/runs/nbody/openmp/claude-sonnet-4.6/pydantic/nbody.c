/* nbody.c — Direct 2-D N-body acceleration with softening, parallelised with OpenMP.
 *
 * For every body i we compute the direct sum over all j != i:
 *   ax[i] = sum_j  dx / d3
 *   ay[i] = sum_j  dy / d3
 * where dx = px[j]-px[i], dy = py[j]-py[i], r2 = dx*dx+dy*dy,
 * d3 = (r2 + EPS2) * sqrt(r2 + EPS2),  EPS2 = 0.0025.
 *
 * The outer loop over i is embarrassingly parallel: each body's accumulator (sx, sy)
 * is private to its iteration, so a plain '#pragma omp parallel for' suffices.
 */

#include <math.h>
#include "nbody.h"

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
