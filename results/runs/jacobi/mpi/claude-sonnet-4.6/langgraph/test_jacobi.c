/* test_jacobi.c — unit test for the parallel Jacobi kernel.
 *
 * Every rank calls jacobi() the same number of times (required for collective
 * operations not to deadlock).  Only rank 0 checks the result against the
 * oracle in reference.h and prints the outcome.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 test_jacobi.c jacobi.o -o test_jacobi -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_jacobi
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

    /* Allocate and fill the initial grid on every rank. */
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

    /* --- collective call: every rank participates ---------------------- */
    jacobi(u, n, iters);

    /* --- only rank 0 checks and reports -------------------------------- */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "jacobi checksum matches oracle");

        /* Verify fixed boundary conditions are intact.
         *
         * The top row is the "hot" boundary: every cell is 1.0.
         * The bottom row and left/right columns are "cold": 0.0.
         * Corner cells belong to the top row first (u_init returns 1.0 for
         * i==0 regardless of j), so top-left and top-right are 1.0.
         */
        int boundary_ok = 1;

        /* Top row: all 1.0 (hot). */
        for (long j = 0; j < n && boundary_ok; j++)
            if (u[0 * n + j] != 1.0) boundary_ok = 0;

        /* Bottom row: all 0.0 (cold). */
        for (long j = 0; j < n && boundary_ok; j++)
            if (u[(n-1) * n + j] != 0.0) boundary_ok = 0;

        /* Left column: rows 1..n-1 are 0.0 (row 0 is hot, already checked). */
        for (long i = 1; i < n && boundary_ok; i++)
            if (u[i * n + 0] != 0.0) boundary_ok = 0;

        /* Right column: rows 1..n-1 are 0.0. */
        for (long i = 1; i < n && boundary_ok; i++)
            if (u[i * n + (n-1)] != 0.0) boundary_ok = 0;

        CHECK(boundary_ok, "fixed boundary rows/cols are unchanged");

        printf("checksum = %.13g  (ref = %.13g, tol = %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);
        int result = TEST_SUMMARY();
        free(u);
        MPI_Finalize();
        return result;
    }

    free(u);
    MPI_Finalize();
    return 0;
}
