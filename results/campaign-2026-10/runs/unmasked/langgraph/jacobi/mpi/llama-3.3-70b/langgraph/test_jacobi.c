#include "jacobi.h"
#include "reference.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *u = malloc(REF_N * REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = (i + j) / (double) REF_N;
        }
    }

    jacobi(u, REF_N, REF_ITERS);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < REF_N; i++) {
            for (long j = 0; j < REF_N; j++) {
                checksum += u[i * REF_N + j];
            }
        }
        if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
            printf("Checksum mismatch: expected %f, got %f\n", REF_CHECKSUM, checksum);
            return 1;
        }
    }

    free(u);
    MPI_Finalize();
    return 0;
}
