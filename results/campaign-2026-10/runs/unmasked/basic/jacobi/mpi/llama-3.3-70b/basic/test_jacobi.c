#include "jacobi.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *u = malloc(REF_N * REF_N * sizeof(double));

    // Initialize u
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = (i + j) * 0.1;
        }
    }

    jacobi(u, REF_N, REF_ITERS);

    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < REF_N; i++) {
            for (long j = 0; j < REF_N; j++) {
                checksum += u[i * REF_N + j];
            }
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL);
        printf("Test passed\n");
    }

    free(u);
    MPI_Finalize();
    return 0;
}
