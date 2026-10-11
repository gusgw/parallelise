#include <mpi.h>
#include <stdlib.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). nbody_accel() is collective, so every
 * rank calls it; only rank 0 checks the checksum against the oracle and reports. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    nbody_accel(px, py, ax, ay, n);

    int rc = 0;
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "nbody acceleration checksum matches the oracle");
        rc = TEST_SUMMARY();
    }
    free(px); free(py); free(ax); free(ay);
    MPI_Finalize();
    return rc;
}
