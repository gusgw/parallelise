#include "kernel_d.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double *x = malloc(REF_M * sizeof(double));
    double *b = malloc(REF_M * sizeof(double));
    for (long i = 0; i < REF_M; i++) {
        b[i] = i;
    }
    cgsolve(x, b, REF_M, REF_ITERS);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < REF_M; i++) {
            checksum += x[i];
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL);
        printf("Test passed!\n");
    }
    MPI_Finalize();
    free(x);
    free(b);
    return 0;
}
