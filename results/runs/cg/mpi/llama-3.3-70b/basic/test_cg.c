#include "cg.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define M_TEST 64
#define ITERS 100

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *b = (double *)malloc(M_TEST * M_TEST * sizeof(double));
    double *x = (double *)malloc(M_TEST * M_TEST * sizeof(double));

    // Initialize b
    for (int i = 0; i < M_TEST * M_TEST; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }

    cgsolve(b, x, M_TEST, ITERS);

    double checksum = 0.0;
    for (int i = 0; i < M_TEST * M_TEST; i++) {
        checksum += x[i];
    }

    if (rank == 0) {
        assert(fabs(checksum - 311942.8986866846) < 10000000);
        printf("Checksum: %f\n", checksum);
    }

    free(b);
    free(x);

    MPI_Finalize();
    return 0;
}
