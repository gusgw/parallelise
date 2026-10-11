#include <mpi.h>
#include <stdlib.h>
#include "kernel_c.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!in || !out) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    kernel_c(in, out, n);

    double chk = checksum(out, n);
    if (rank == 0) {
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");
    }

    free(in);
    free(out);
    MPI_Finalize();
    return TEST_SUMMARY();
}
