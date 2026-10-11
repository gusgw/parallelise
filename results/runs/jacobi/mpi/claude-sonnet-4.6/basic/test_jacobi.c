/* test_jacobi.c — unit test for the parallel Jacobi kernel.
 *
 * Every rank calls jacobi() with the same initial grid.  After the call,
 * rank 0 computes the checksum and checks it against the oracle in
 * reference.h within REF_TOL.  All ranks participate in every collective
 * call so mpirun never hangs.
 */

#include <mpi.h>
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

    /* Allocate and fill the initial grid (every rank does this). */
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

    /* Run the collective kernel (all ranks participate). */
    jacobi(u, n, iters);

    /* Rank 0 validates the result. */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "jacobi checksum matches oracle");
        printf("checksum = %.13g  (ref = %.13g, tol = %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);
    }

    free(u);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    /* Broadcast the exit code so all ranks return the same value. */
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return result;
}
