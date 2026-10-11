#include "nbody.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#define EPS2 0.0025

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Decompose bodies i across ranks: each rank computes ax[i], ay[i] for its block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_count = hi - lo;

    /* Temporary arrays for local results */
    double *local_ax = malloc(local_count * sizeof(double));
    double *local_ay = malloc(local_count * sizeof(double));

    /* Compute accelerations for local block */
    for (long i = lo; i < hi; i++) {
        double sx = 0.0;
        double sy = 0.0;
        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        local_ax[i - lo] = sx;
        local_ay[i - lo] = sy;
    }

    /* Gather counts from all ranks */
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)(r_hi - r_lo);
    }
    
    displs[0] = 0;
    for (int r = 1; r < size; r++) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    /* Allgatherv to combine results: each rank gets full ax and ay */
    MPI_Allgatherv(local_ax, (int)local_count, MPI_DOUBLE,
                   ax, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_ay, (int)local_count, MPI_DOUBLE,
                   ay, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_ax);
    free(local_ay);
    free(recvcounts);
    free(displs);
}
