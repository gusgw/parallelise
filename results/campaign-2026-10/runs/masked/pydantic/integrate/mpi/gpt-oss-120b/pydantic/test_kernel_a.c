#include <mpi.h>
#include <stdio.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Call the kernel once (all ranks must call the same number of times)
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        if (tests_failed == 0) {
            printf("PASS: kernel_a matches reference within tolerance\n");
        }
    }

    MPI_Finalize();
    return TEST_SUMMARY();
}
