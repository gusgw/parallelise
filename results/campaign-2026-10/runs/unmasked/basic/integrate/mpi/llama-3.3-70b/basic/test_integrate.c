#include "integrate.h"
#include "reference.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double result = integrate(REF_A, REF_B, REF_N_TEST);
    if (rank == 0) {
        assert(fabs(result - REF_EXPECTED) < REF_TOL);
        printf("Result: %f\n", result);
    }
    MPI_Finalize();
    return 0;
}
