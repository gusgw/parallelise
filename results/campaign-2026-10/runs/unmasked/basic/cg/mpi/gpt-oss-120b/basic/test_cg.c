#include <mpi.h>
#include <stdlib.h>
#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    for (long idx = 0; idx < n; ++idx) {
        b[idx] = b_of(idx);
    }

    cgsolve(x, b, m, iters);

    if (rank == 0) {
        double chk = checksum(x, m);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");
    }

    free(b);
    free(x);
    MPI_Finalize();
    return TEST_SUMMARY();
}
