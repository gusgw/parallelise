#include "nbody.h"
#include "reference.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *px = (double *)malloc(REF_N * sizeof(double));
    double *py = (double *)malloc(REF_N * sizeof(double));
    double *ax = (double *)malloc(REF_N * sizeof(double));
    double *ay = (double *)malloc(REF_N * sizeof(double));

    // Fill px and py with deterministic positions in [-1, 1)
    for (long i = 0; i < REF_N; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }

    // Call the kernel the same number of times on every rank
    for (int i = 0; i < 10; i++)
        nbody_accel(px, py, ax, ay, REF_N);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        double checksum = 0.0;
        for (long i = 0; i < REF_N; i++)
            checksum += fabs(ax[i]) + fabs(ay[i]);
        printf("checksum = %f\n", checksum);
        if (fabs(checksum - REF_CHECKSUM) > REF_TOL)
            printf("ERROR: checksum mismatch\n");
        else
            printf("OK\n");
    }

    free(px);
    free(py);
    free(ax);
    free(ay);

    MPI_Finalize();
    return 0;
}
