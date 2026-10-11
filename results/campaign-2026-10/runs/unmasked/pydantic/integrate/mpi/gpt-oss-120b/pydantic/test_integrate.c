#include <mpi.h>
#include <stdio.h>
#include "integrate.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // All ranks compute the integral
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "integrate result");
        printf("Result = %.15g\n", result);
    }

    MPI_Finalize();
    return TEST_SUMMARY();
}
