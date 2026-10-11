#include <mpi.h>
#include <stdlib.h>
#include "kernel_b.h"
#include "reference.h"
#include "testing.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int iters = REF_ITERS;
    // allocate output array on all ranks (will be filled by kernel)
    double *u = (double *)malloc(n * n * sizeof(double));
    if (!u) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize output buffer to zero (not strictly needed, but safe)
    for (long i = 0; i < n * n; ++i) u[i] = 0.0;

    // Call the kernel (all ranks must call the same number of times)
    kernel_b(u, n, iters);

    // Only rank 0 checks the result against reference checksum
    if (rank == 0) {
        double chk = checksum(u, n);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(u);
    MPI_Finalize();
    return TEST_SUMMARY();
}
