#include "nbody.h"
#include <math.h>
#include <omp.h>

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    const double EPS2 = 0.0025;

    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double sx = 0.0;
        double sy = 0.0;
        double pix = px[i];
        double piy = py[i];

        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - pix;
            double dy = py[j] - piy;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}
