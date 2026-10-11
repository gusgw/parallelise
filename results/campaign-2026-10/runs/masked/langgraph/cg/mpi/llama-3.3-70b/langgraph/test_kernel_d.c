#include "kernel_d.h"
#include "reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        printf("Checksum: %f\n", checksum);
        if (fabs(checksum - REF_CHECKSUM) < REF_TOL) {
            printf("Test passed!\n");
        } else {
            printf("Test failed!\n");
        }
    }

    free(x);
    free(b);

    MPI_Finalize();
    return 0;
}
