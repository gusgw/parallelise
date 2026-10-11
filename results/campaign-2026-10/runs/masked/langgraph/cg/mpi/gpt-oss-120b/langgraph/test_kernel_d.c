#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long m = REF_M;
    int iters = (int)REF_ITERS;

    /* Allocate vectors */
    double *x = (double *)malloc(sizeof(double) * m * m);
    double *b = (double *)malloc(sizeof(double) * m * m);
    if (!x || !b) {
        fprintf(stderr, "allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill b using helper */
    for (long i = 0; i < m * m; ++i) b[i] = b_of(i);

    /* Every rank runs the kernel the same number of times */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double chk = checksum(x, m);
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(x);
    free(b);
    MPI_Finalize();
    return TEST_SUMMARY();
}
