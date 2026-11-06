/* The serial routine of nbody.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 30, 300, 2000, 16000]. */
#include "nbody_accel.h"
#include <math.h>

#define SOFTENING_SQUARED 0.0025
#define BLOCK_SIZE 256

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    for (long start = 0; start < n; start += BLOCK_SIZE) {
        long end = start + BLOCK_SIZE;
        if (end > n) {
            end = n;
        }
        
        for (long i = start; i < end; i++) {
            double sum_ax = 0.0;
            double sum_ay = 0.0;
            
            for (long j = 0; j < n; j++) {
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double r2 = dx * dx + dy * dy + SOFTENING_SQUARED;
                double d3 = r2 * sqrt(r2);
                sum_ax += dx / d3;
                sum_ay += dy / d3;
            }
            
            ax[i] = sum_ax;
            ay[i] = sum_ay;
        }
    }
}
