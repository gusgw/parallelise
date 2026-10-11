#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "matmul.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    /* Allocate and fill matrices A and B (full copies on all ranks) */
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    if (!A || !B || !C) {
        if (rank == 0) fprintf(stderr, "Memory allocation failed\n");
        MPI_Finalize();
        return 1;
    }

    /* Fill A and B deterministically */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Call the collective kernel (all ranks call it) */
    matmul(A, B, C, n);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(C, n);
        double err = fabs(cs - REF_CHECKSUM);
        printf("checksum = %.1f, expected = %.1f, error = %.6f, tol = %.1f\n",
               cs, REF_CHECKSUM, err, REF_TOL);
        if (err > REF_TOL) {
            printf("FAIL: error exceeds tolerance\n");
            free(A);
            free(B);
            free(C);
            MPI_Finalize();
            return 1;
        }
        printf("PASS\n");
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
