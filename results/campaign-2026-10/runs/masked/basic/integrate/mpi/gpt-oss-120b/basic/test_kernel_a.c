#include <mpi.h>
#include <stdio.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Call the kernel; all ranks must invoke it the same number of times. */
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        if (tests_failed == 0) {
            printf("kernel_a passed. result = %.15g\n", result);
        }
    }

    MPI_Finalize();
    return TEST_SUMMARY();
}
