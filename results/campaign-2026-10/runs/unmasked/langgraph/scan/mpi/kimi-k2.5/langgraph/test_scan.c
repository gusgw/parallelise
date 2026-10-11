#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

    /* Allocate full arrays (each rank has a full copy of input) */
    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));

    /* Fill input deterministically */
    for (long i = 0; i < REF_N; i++) {
        in[i] = in_of(i);
    }

    /* Call the collective kernel */
    scan(in, out, REF_N);

    /* Gather full output to rank 0 for validation */
    /* First, compute local block sizes */
    long lo = REF_N * rank / size;
    long hi = REF_N * (rank + 1) / size;
    long local_n = hi - lo;

    /* Prepare receive counts and displacements for MPI_Gatherv */
    int *recvcounts = NULL;
    int *displs = NULL;
    double *full_out = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            long r_lo = REF_N * r / size;
            long r_hi = REF_N * (r + 1) / size;
            recvcounts[r] = (int)(r_hi - r_lo);
            displs[r] = (int)r_lo;
        }
        full_out = (double *)malloc(REF_N * sizeof(double));
    }

    /* Gather all blocks to rank 0 */
    MPI_Gatherv(out + lo, (int)local_n, MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 validates and reports */
    if (rank == 0) {
        double cs = checksum(full_out, REF_N);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int result = TEST_SUMMARY();
        free(recvcounts);
        free(displs);
        free(full_out);
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
