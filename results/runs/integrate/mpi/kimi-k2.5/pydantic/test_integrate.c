#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "integrate.h"
#include "reference.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* All ranks call the kernel (collective operation) */
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    /* Only rank 0 checks and prints the result */
    if (rank == 0) {
        double error = fabs(result - REF_EXPECTED);
        printf("result   = %.15f\n", result);
        printf("expected = %.15f\n", REF_EXPECTED);
        printf("error    = %.15e\n", error);
        printf("tol      = %.15e\n", REF_TOL);

        if (error > REF_TOL) {
            printf("FAIL: error exceeds tolerance\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        printf("PASS\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
