/*
 * test_jacobi.c — unit test for the parallel Jacobi kernel.
 *
 * Every rank calls jacobi() the same number of times (once) so MPI does not
 * deadlock.  Only rank 0 checks the result against the oracle in reference.h
 * and prints the outcome.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 test_jacobi.c jacobi.o -o test_jacobi -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_jacobi
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Allocate and initialise the full grid on every rank. */
    long n     = REF_N;
    int  iters = (int)REF_ITERS;

    double *u = (double *)malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);

    /* Every rank participates in the collective kernel. */
    jacobi(u, n, iters);

    /* Only rank 0 validates and prints. */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "jacobi checksum matches oracle");
        printf("checksum = %.13g  (ref = %.13g, tol = %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);
        /* Return value of TEST_SUMMARY() is used as exit code below. */
        int result = TEST_SUMMARY();
        free(u);
        MPI_Finalize();
        return result;
    }

    free(u);
    MPI_Finalize();
    return 0;
}
