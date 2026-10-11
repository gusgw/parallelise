#include <mpi.h>
#include <stdio.h>
#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* All ranks call the kernel (collective operation) */
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    /* Only rank 0 checks and prints results */
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate result");
        printf("integrate(%g, %g, %ld) = %.12g\n", REF_A, REF_B, REF_N_TEST, result);
        printf("expected = %.12g, error = %g\n", REF_EXPECTED, result - REF_EXPECTED);
    }

    int failed = 0;
    if (rank == 0) {
        failed = TEST_SUMMARY();
    }

    /* Broadcast the test result to all ranks so they all exit with same status */
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return failed;
}
