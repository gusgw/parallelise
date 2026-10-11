/* Unit test for the parallel CG kernel */

#include <mpi.h>
#include <stdlib.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    /* Allocate full RHS and solution vectors (every rank holds full copy) */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    for (long i = 0; i < n; ++i) {
        b[i] = b_of(i);
        x[i] = 0.0;
    }

    /* Call the kernel – every rank must call it the same number of times */
    cgsolve(x, b, m, iters);

    /* Verify checksum on rank 0 */
    if (rank == 0) {
        double chk = checksum(x, m);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");
    }

    free(b);
    free(x);
    MPI_Finalize();
    return TEST_SUMMARY();
}
