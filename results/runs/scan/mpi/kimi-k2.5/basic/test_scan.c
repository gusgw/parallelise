#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Allocate full arrays (each rank holds a full copy of input) */
    double *in = malloc(REF_N * sizeof(double));
    double *out = malloc(REF_N * sizeof(double));

    /* Fill input deterministically (every rank does the same) */
    for (long i = 0; i < REF_N; i++) {
        in[i] = in_of(i);
    }

    /* Call the collective kernel */
    scan(in, out, REF_N);

    /* Gather full output on rank 0 for verification */
    double *full_out = NULL;
    if (rank == 0) {
        full_out = malloc(REF_N * sizeof(double));
    }

    /* Compute this rank's block */
    long lo = REF_N * rank / size;
    long hi = REF_N * (rank + 1) / size;
    long count = hi - lo;

    /* Gather counts from all ranks */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    int mycount = (int)count;
    MPI_Gather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            long r_lo = REF_N * i / size;
            displs[i] = (int)r_lo;
        }
    }

    MPI_Gatherv(out + lo, mycount, MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 verifies the result */
    if (rank == 0) {
        double cs = checksum(full_out, REF_N);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

        int result = TEST_SUMMARY();
        free(full_out);
        free(recvcounts);
        free(displs);
        free(in);
        free(out);
        MPI_Finalize();
        return result;
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
