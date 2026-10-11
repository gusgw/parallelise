#include "cg.h"
#include "reference.h"
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int m = 64;
    int iters = 100;
    double *b = (double *)malloc(m * m * sizeof(double));
    for (int i = 0; i < m * m; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }
    double *x = (double *)malloc(m * m * sizeof(double));
    cgsolve(x, b, m, iters);
    if (rank == 0) {
        double checksum = 0.0;
        for (int i = 0; i < m * m; i++) {
            checksum += x[i];
        }
        if (fabs(checksum - 272069.953528) > 1e-3) {
            printf("Checksum mismatch: %f vs %f\n", checksum, 272069.953528);
            return 1;
        }
    }
    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}
