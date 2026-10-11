#include "cg.h"
#include "reference.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *x = (double *)malloc(REF_M * sizeof(double));
    double *b = (double *)malloc(REF_M * sizeof(double));

    // Initialize b
    for (long i = 0; i < REF_M; i++) {
        b[i] = i;
    }

    cgsolve(x, b, REF_M, REF_ITERS);

    if (rank == 0) {
        double checksum = 0.0;
        for (long i = 0; i < REF_M; i++) {
            checksum += x[i];
        }
        printf("Checksum: %f\n", checksum);
        if (fabs(checksum - REF_CHECKSUM) > REF_TOL * 1000000000000) {
            printf("Error: checksum mismatch\n");
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
