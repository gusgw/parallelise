#include "nbody.h"
#include "mpi.h"
#include <math.h>

void nbody(double *px, double *py, double *ax, double *ay, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int lo = n * rank / size;
    int hi = n * (rank + 1) / size;

    for (int i = lo; i < hi; i++) {
        double sx = 0.0;
        double sy = 0.0;
        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + 0.0025) * sqrt(r2 + 0.0025);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}