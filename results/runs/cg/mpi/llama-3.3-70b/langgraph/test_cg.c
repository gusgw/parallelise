#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = M_TEST;
    int iters = ITERS;
    double *b = (double *)malloc(m * m * sizeof(double));
    double *x = (double *)malloc(m * m * sizeof(double));

    // Initialize b
    for (int i = 0; i < m * m; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }

    // Solve the system
    cgsolve(x, b, m, iters);

    // Compute the checksum
    double checksum = 0.0;
    for (int i = 0; i < m * m; i++) {
        checksum += x[i];
    }

    if (rank == 0) {
        printf("Checksum: %f\n", checksum);
        if (fabs(checksum - CHECKSUM) > REF_TOL) {
            printf("Error: checksum does not match reference\n");
            return 1;
        }
    }

    free(b);
    free(x);

    MPI_Finalize();
    return 0;
}
