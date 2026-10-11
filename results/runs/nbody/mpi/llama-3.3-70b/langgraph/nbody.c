#include "nbody.h"
#include <math.h>
#include <mpi.h>

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    int i, j, rank, size;
    double dx, dy, r2, d3, sx, sy;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long start = n * rank / size;
    long end = n * (rank + 1) / size;
    for (i = start; i < end; i++) {
        sx = 0.0;
        sy = 0.0;
        for (j = 0; j < n; j++) {
            if (j == i)
                continue;
            dx = px[j] - px[i];
            dy = py[j] - py[i];
            r2 = dx * dx + dy * dy;
            d3 = (r2 + 0.0025) * sqrt(r2 + 0.0025);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}