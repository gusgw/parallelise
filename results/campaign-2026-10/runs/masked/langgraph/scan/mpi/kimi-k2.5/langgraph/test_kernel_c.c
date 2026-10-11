#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_c.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!in || !out) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* All ranks call the kernel (collective operation) */
    kernel_c(in, out, n);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int failed = TEST_SUMMARY();
        free(in);
        free(out);
        MPI_Finalize();
        return failed;
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
