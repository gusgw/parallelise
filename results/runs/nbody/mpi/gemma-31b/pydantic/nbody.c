#include "nbody.h"
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = (n * rank) / size;
    long hi = (n * (rank + 1)) / size;
    long count = hi - lo;

    double *local_ax = (double *)malloc(count * sizeof(double));
    double *local_ay = (double *)malloc(count * sizeof(double));

    const double EPS2 = 0.0025;

    for (long i = lo; i < hi; i++) {
        double sx = 0.0;
        double sy = 0.0;
        double px_i = px[i];
        double py_i = py[i];
        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - px_i;
            double dy = py[j] - py_i;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        local_ax[i - lo] = sx;
        local_ay[i - lo] = sy;
    }

    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long r_lo = (n * r) / size;
        long r_hi = (n * (r + 1)) / size;
        recvcounts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }

    MPI_Allgatherv(local_ax, count, MPI_DOUBLE, ax, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_ay, count, MPI_DOUBLE, ay, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_ax);
    free(local_ay);
    free(recvcounts);
    free(displs);
}
