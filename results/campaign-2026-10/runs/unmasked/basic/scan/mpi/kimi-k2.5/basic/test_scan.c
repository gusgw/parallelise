#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!in || !out) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill input deterministically */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Call the collective kernel */
    scan(in, out, n);

    /* Gather all blocks to rank 0 for verification */
    double *full_out = NULL;
    if (rank == 0) {
        full_out = malloc(sizeof(double) * n);
        if (!full_out) {
            fprintf(stderr, "alloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Compute local block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Gather sizes from all ranks */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        for (int r = 0; r < size; r++) {
            long r_lo = n * r / size;
            long r_hi = n * (r + 1) / size;
            recvcounts[r] = (int)(r_hi - r_lo);
            displs[r] = (int)r_lo;
        }
    }

    /* Gather the distributed output to rank 0 */
    MPI_Gatherv(out + lo, (int)local_n, MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 verifies and reports */
    int result = 0;
    if (rank == 0) {
        double cs = checksum(full_out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        result = TEST_SUMMARY();
        free(full_out);
        free(recvcounts);
        free(displs);
    }

    free(in);
    free(out);
    MPI_Finalize();
    return result;
}
