#include <math.h>

#include "nbody.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Each body's
 * accelerations depend only on the read-only positions, so the outer loop over i
 * parallelises directly. The inner loop keeps the notebook's j order so the rounding
 * matches the oracle. */
void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n)
{
    const double eps2 = 0.0025;   /* softening 0.05 squared */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
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
}
