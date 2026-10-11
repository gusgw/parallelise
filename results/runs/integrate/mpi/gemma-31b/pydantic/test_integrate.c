#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "integrate.h"
#include "reference.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    
    if (rank == 0) {
        double diff = fabs(result - REF_EXPECTED);
        printf("Result: %.15f, Expected: %.15f, Diff: %.15f\n", result, REF_EXPECTED, diff);
        if (diff <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
