#include <mpi.h>
#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). cgsolve() is collective, so every rank
 * calls it exactly once (the same number of times on every rank -- unbalanced calls
 * would hang in the collectives); only rank 0 checks the checksum against the oracle
 * and reports. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    for (long t = 0; t < m * m; t++) b[t] = b_of(t);

    cgsolve(x, b, m, REF_ITERS);

    int rc = 0;
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "cg checksum matches the oracle");
        rc = TEST_SUMMARY();
    }
    free(x); free(b);
    MPI_Finalize();
    return rc;
}
