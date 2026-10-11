/* test_nbody.c — unit test for the nbody_accel kernel.
 *
 * Every rank calls nbody_accel the same number of times (once) so that the
 * collective MPI operations inside the kernel do not deadlock.  Only rank 0
 * checks the result against the oracle in reference.h and prints the outcome.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 test_nbody.c nbody.o -o test_nbody -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_nbody
 */

#include <mpi.h>
#include <math.h>
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

    /* Allocate and fill positions on every rank (identical deterministic fill) */
    double *px = (double *)malloc(sizeof(double) * n);
    double *py = (double *)malloc(sizeof(double) * n);
    double *ax = (double *)malloc(sizeof(double) * n);
    double *ay = (double *)malloc(sizeof(double) * n);
    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Collective call — every rank participates */
    nbody_accel(px, py, ax, ay, n);

    /* Only rank 0 validates and prints */
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL * REF_CHECKSUM,
                    "nbody checksum matches oracle");
    }

    free(px); free(py); free(ax); free(ay);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks exit consistently */
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return result;
}
