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
        double actual_checksum = checksum(C, n);
        printf("Expected checksum: %f\n", REF_CHECKSUM);
        printf("Actual checksum:   %f\n", actual_checksum);
        if (fabs(actual_checksum - REF_CHECKSUM) <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            free(A);
            free(B);
            free(C);
            MPI_Finalize();
            return 1;
        }
    }

    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}
