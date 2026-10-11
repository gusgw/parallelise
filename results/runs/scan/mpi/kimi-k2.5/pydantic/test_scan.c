#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "scan.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;

    /* Allocate full input (every rank has a copy) */
    double *in = malloc(n * sizeof(double));
    if (!in) {
        fprintf(stderr, "Failed to allocate input\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill input deterministically */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Compute this rank's output block size */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long count = hi - lo;

    /* Allocate output for this rank's block */
    double *out = malloc(count * sizeof(double));
    if (!out) {
        fprintf(stderr, "Failed to allocate output\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Call the kernel (collective) */
    scan(in, out, n);

    /* Gather all output blocks to rank 0 for verification */
    double *full_out = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        full_out = malloc(n * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if (!full_out || !recvcounts || !displs) {
            fprintf(stderr, "Failed to allocate gather buffers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Prepare gather parameters on all ranks */
    int my_count = (int)count;
    int my_displ = (int)lo;

    /* Gather counts and displacements on rank 0 */
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_displ, 1, MPI_INT, displs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Gather the output blocks */
    MPI_Gatherv(out, my_count, MPI_DOUBLE, full_out, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Rank 0 verifies the result */
    if (rank == 0) {
        double cs = checksum(full_out, n);
        double err = fabs(cs - REF_CHECKSUM);

        printf("checksum = %.16f\n", cs);
        printf("expected = %.16f\n", REF_CHECKSUM);
        printf("error    = %.16e\n", err);
        printf("tol      = %.16e\n", REF_TOL);

        if (err <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
        }
    }

    /* Cleanup */
    free(in);
    free(out);
    if (rank == 0) {
        free(full_out);
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
