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

    /* Decompose bodies i across ranks: each rank computes ax/ay for its block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long count = hi - lo;

    /* Allocate temporary arrays for this rank's partial results */
    double *local_ax = (double *)malloc(count * sizeof(double));
    double *local_ay = (double *)malloc(count * sizeof(double));

    /* Compute accelerations for bodies [lo, hi) */
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

    /* Gather counts from all ranks - all ranks need recvcounts and displs for Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int local_count = (int)count;
    MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    /* Compute displacements */
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    /* Allgatherv to combine all partial results */
    MPI_Allgatherv(local_ax, local_count, MPI_DOUBLE,
                   ax, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(local_ay, local_count, MPI_DOUBLE,
                   ay, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local_ax);
    free(local_ay);
    free(recvcounts);
    free(displs);
}
