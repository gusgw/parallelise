#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *px = malloc(n * sizeof(double));
    double *py = malloc(n * sizeof(double));
    double *ax = malloc(n * sizeof(double));
    double *ay = malloc(n * sizeof(double));

    /* Fill positions deterministically (same on all ranks) */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Collective kernel: all ranks call it together */
    nbody_accel(px, py, ax, ay, n);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(px);
    free(py);
    free(ax);
    free(ay);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    MPI_Finalize();
    return result;
}
