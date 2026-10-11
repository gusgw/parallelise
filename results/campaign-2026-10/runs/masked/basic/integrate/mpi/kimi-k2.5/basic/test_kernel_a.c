#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* All ranks call the kernel (collective operation) */
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);

    /* Only rank 0 checks and reports results */
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        int status = TEST_SUMMARY();
        MPI_Finalize();
        return status;
    }

    MPI_Finalize();
    return 0;
}
