#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "integrate.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* All ranks call the kernel collectively */
    double result = integrate(REF_A, REF_B, REF_N_TEST);

    /* Only rank 0 checks and reports results */
    if (rank == 0) {
        double err = fabs(result - REF_EXPECTED);
        printf("integrate(%g, %g, %ld) = %.15g\n", REF_A, REF_B, REF_N_TEST, result);
        printf("expected = %.15g\n", REF_EXPECTED);
        printf("error = %g (tol = %g)\n", err, REF_TOL);

        if (err > REF_TOL) {
            printf("FAIL: error exceeds tolerance\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        printf("PASS\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
