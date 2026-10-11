#include "nbody.h"
#include <mpi.h>
#include <math.h>
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

    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    int local_n = (int)count;
    MPI_Allgather(&local_n, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }

    MPI_Allgatherv(local_ax, (int)count, MPI_DOUBLE, ax, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_ay, (int)count, MPI_DOUBLE, ay, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    free(local_ax);
    free(local_ay);
    free(recv_counts);
    free(displs);
}
