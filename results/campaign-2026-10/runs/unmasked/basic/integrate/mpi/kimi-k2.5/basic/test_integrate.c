#include <mpi.h>
#include <stdio.h>
#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* All ranks call the kernel collectively */
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    /* Only rank 0 checks and reports results */
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate result");
        int exit_code = TEST_SUMMARY();
        MPI_Finalize();
        return exit_code;
    }

    MPI_Finalize();
    return 0;
}
