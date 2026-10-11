#include "integrate.h"
#include "reference.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < 10; i++) {
        double approx = integrate(REF_A, REF_B, REF_N_TEST);
        if (rank == 0) {
            assert(fabs(approx - REF_EXPECTED) <= REF_TOL);
            printf("approx = %f\n", approx);
        }
    }
    MPI_Finalize();
    return 0;
}
