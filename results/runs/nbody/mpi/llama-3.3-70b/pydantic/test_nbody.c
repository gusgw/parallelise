#include "nbody.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *px = (double *)malloc(1024 * sizeof(double));
    double *py = (double *)malloc(1024 * sizeof(double));
    double *ax = (double *)malloc(1024 * sizeof(double));
    double *ay = (double *)malloc(1024 * sizeof(double));

    for (int i = 0; i < 1024; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }

    nbody(px, py, ax, ay, 1024);

    double checksum = 0.0;
    for (int i = 0; i < 1024; i++) {
        checksum += fabs(ax[i]) + fabs(ay[i]);
    }

    if (rank == 0) {
        printf("checksum = %f\n", checksum);
        if (fabs(checksum - 1053395.4952812886) > 1e-6)
            printf("ERROR: checksum mismatch\n");
        else
            printf("PASSED\n");
    }

    MPI_Finalize();
    return 0;
}