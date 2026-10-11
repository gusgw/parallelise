#include "kernel_b.h"
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
    double *u = malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        u[i] = i;
    }
    kernel_b(u, REF_N, REF_ITERS);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < REF_N; i++) {
            checksum += u[i];
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL);
        printf("Test passed\n");
    }
    MPI_Finalize();
    free(u);
    return 0;
}
