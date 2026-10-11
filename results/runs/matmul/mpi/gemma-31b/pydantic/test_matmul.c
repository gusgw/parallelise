#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    matmul(A, B, C, n);

    if (rank == 0) {
        double sum = checksum(C, n);
        printf("Checksum: %f, Expected: %f, Diff: %f\n", sum, REF_CHECKSUM, fabs(sum - REF_CHECKSUM));
        if (fabs(sum - REF_CHECKSUM) <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
