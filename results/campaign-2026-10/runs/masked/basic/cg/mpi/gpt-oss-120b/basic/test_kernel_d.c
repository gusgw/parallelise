#include <mpi.h>
#include <stdlib.h>
#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    long n = m * m;
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    // Fill b using the provided helper
    for (long i = 0; i < n; ++i) {
        b[i] = b_of(i);
    }

    // Solve
    cgsolve(x, b, m, (int)REF_ITERS);

    // Compute checksum on rank 0
    if (rank == 0) {
        double chk = checksum(x, m);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(b);
    free(x);
    MPI_Finalize();
    return TEST_SUMMARY();
}
