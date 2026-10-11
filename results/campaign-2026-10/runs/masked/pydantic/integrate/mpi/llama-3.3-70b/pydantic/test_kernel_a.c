#include "kernel_a.h"
#include "reference.h"
#include "mpi.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double result;
    for (int i = 0; i < 10; i++) {
        result = kernel_a(REF_A, REF_B, REF_N_TEST);
    }
    if (rank == 0) {
        assert(fabs(result - REF_EXPECTED) < REF_TOL);
        printf("Result: %f\n", result);
    }
    MPI_Finalize();
    return 0;
}
