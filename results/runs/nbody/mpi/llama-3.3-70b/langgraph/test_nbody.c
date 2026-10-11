#include "nbody.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long n = 1024;
    double *px, *py, *ax, *ay;
    double checksum;
    long i;
    MPI_Init(&argc, &argv);
    px = (double *)malloc(n * sizeof(double));
    py = (double *)malloc(n * sizeof(double));
    ax = (double *)malloc(n * sizeof(double));
    ay = (double *)malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        px[i] = ((i * 73) % 4093) / 4093.0 * 2.0 - 1.0;
        py[i] = ((i * 151) % 4093) / 4093.0 * 2.0 - 1.0;
    }
    nbody_accel(px, py, ax, ay, n);
    checksum = 0.0;
    for (i = 0; i < n; i++)
        checksum += fabs(ax[i]) + fabs(ay[i]);
    if (fabs(checksum - 1053395.4952812886) > 1e-6)
        printf("Error: checksum mismatch\n");
    else
        printf("OK\n");
    MPI_Finalize();
    return 0;
}