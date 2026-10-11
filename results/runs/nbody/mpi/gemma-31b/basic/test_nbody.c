#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include "nbody.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    double *px = (double *)malloc(n * sizeof(double));
    double *py = (double *)malloc(n * sizeof(double));
    double *ax = (double *)malloc(n * sizeof(double));
    double *ay = (double *)malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        printf("Checksum: %f (Expected: %f, Tol: %f)\n", cs, REF_CHECKSUM, REF_TOL);
        assert(fabs(cs - REF_CHECKSUM) < REF_TOL);
        printf("Test passed!\n");
    }

    free(px);
    free(py);
    free(ax);
    free(ay);

    MPI_Finalize();
    return 0;
}
