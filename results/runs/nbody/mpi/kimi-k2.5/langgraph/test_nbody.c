#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nbody.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    /* Allocate arrays */
    double *px = malloc(n * sizeof(double));
    double *py = malloc(n * sizeof(double));
    double *ax = malloc(n * sizeof(double));
    double *ay = malloc(n * sizeof(double));

    /* Fill positions deterministically (all ranks do the same) */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Call the kernel (collective) */
    nbody_accel(px, py, ax, ay, n);

    /* Compute checksum and verify (only rank 0) */
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int result = TEST_SUMMARY();
        free(px);
        free(py);
        free(ax);
        free(ay);
        MPI_Finalize();
        return result;
    }

    free(px);
    free(py);
    free(ax);
    free(ay);
    MPI_Finalize();
    return 0;
}
