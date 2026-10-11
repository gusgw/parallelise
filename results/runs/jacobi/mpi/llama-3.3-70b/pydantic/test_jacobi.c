#include "jacobi.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *u = (double *) malloc(REF_N * REF_N * sizeof(double));

    // initialize grid
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = (i == 0) ? 1.0 : 0.0;
        }
    }

    jacobi(u, REF_N, REF_ITERS);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        double checksum = 0.0;
        for (long i = 0; i < REF_N; i++) {
            for (long j = 0; j < REF_N; j++) {
                checksum += u[i * REF_N + j];
            }
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL);
        printf("checksum = %f\n", checksum);
    }

    MPI_Finalize();
    return 0;
}
