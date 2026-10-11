#include <mpi.h>
#include <stdio.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Compute the integral using the parallel kernel.
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);

    // Only rank 0 performs the check.
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        if (tests_failed == 0) {
            printf("PASS: kernel_a result = %.12g\n", result);
        }
    }

    MPI_Finalize();
    return TEST_SUMMARY();
}
