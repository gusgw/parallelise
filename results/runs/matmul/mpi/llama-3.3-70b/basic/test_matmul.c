#include "matmul.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *A = (double *)malloc(REF_N * REF_N * sizeof(double));
    double *B = (double *)malloc(REF_N * REF_N * sizeof(double));
    double *C = (double *)malloc(REF_N * REF_N * sizeof(double));

    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            A[i * REF_N + j] = ((i * 3 + j * 7) % 11) + 1;
            B[i * REF_N + j] = ((i * 5 + j * 2) % 13) + 1;
        }
    }

    matmul(A, B, C, REF_N);

    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            checksum += C[i * REF_N + j];
        }
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        assert(fabs(checksum - REF_CHECKSUM) <= REF_TOL);
        printf("checksum = %f\n", checksum);
    }

    MPI_Finalize();
    return 0;
}
