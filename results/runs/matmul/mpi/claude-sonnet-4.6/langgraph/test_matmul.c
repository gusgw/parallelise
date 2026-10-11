/* test_matmul.c -- unit test for the parallel matmul kernel.
 *
 * Every rank calls matmul() the same number of times (once) so MPI never
 * hangs.  Only rank 0 checks the checksum against the reference oracle and
 * prints the result.
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    /* Every rank allocates and fills A, B, and C */
    double *A = (double *)malloc(sizeof(double) * n * n);
    double *B = (double *)malloc(sizeof(double) * n * n);
    double *C = (double *)malloc(sizeof(double) * n * n);
    if (!A || !B || !C) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Collective call -- every rank participates */
    matmul(A, B, C, n);

    /* Only rank 0 validates */
    if (rank == 0) {
        double cs = checksum(C, n);
        double err = fabs(cs - REF_CHECKSUM);
        if (err <= REF_TOL) {
            printf("PASS  checksum=%.1f  ref=%.1f  err=%.6g\n",
                   cs, REF_CHECKSUM, err);
        } else {
            printf("FAIL  checksum=%.1f  ref=%.1f  err=%.6g  tol=%.6g\n",
                   cs, REF_CHECKSUM, err, REF_TOL);
            free(A); free(B); free(C);
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
