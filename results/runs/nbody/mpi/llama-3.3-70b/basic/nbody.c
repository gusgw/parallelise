#include "nbody.h"
#include "reference.h"
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    for (long i = lo; i < hi; i++) {
        double sx = 0.0;
        double sy = 0.0;
        for (long j = 0; j < n; j++) {
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

    // Assemble the full ax and ay on every rank with a gather
    double *ax_gather = (double *)malloc(n * sizeof(double));
    double *ay_gather = (double *)malloc(n * sizeof(double));
    MPI_Allgather(ax + lo, hi - lo, MPI_DOUBLE, ax_gather, hi - lo, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(ay + lo, hi - lo, MPI_DOUBLE, ay_gather, hi - lo, MPI_DOUBLE, MPI_COMM_WORLD);

    // Copy the gathered values back to ax and ay
    for (long i = 0; i < n; i++) {
        ax[i] = ax_gather[i];
        ay[i] = ay_gather[i];
    }

    free(ax_gather);
    free(ay_gather);
}
