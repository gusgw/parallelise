#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;

    /* Allocate input and output arrays (each rank holds full copy of input) */
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));

    /* Fill input using deterministic pseudo-random fill */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Call the kernel (collective) */
    scan(in, out, n);

    /* Compute this rank's block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Gather all blocks to rank 0 for verification */
    double *full_out = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        full_out = (double *)malloc(n * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        for (int r = 0; r < size; r++) {
            long r_lo = n * r / size;
            long r_hi = n * (r + 1) / size;
            recvcounts[r] = (int)(r_hi - r_lo);
            displs[r] = (int)r_lo;
        }
    }

    /* Gather the distributed output to rank 0 */
    MPI_Gatherv(&out[lo], (int)(hi - lo), MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 verifies and reports */
    if (rank == 0) {
        /* Compute checksum of the full output */
        double cs = checksum(full_out, n);

        /* Check against reference */
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");

        /* Print result */
        printf("checksum = %.12g (expected %.12g, tol %g)\n", cs, REF_CHECKSUM, REF_TOL);

        free(full_out);
        free(recvcounts);
        free(displs);
    }

    free(in);
    free(out);

    int result = 0;
    if (rank == 0) {
        result = TEST_SUMMARY();
    }

    MPI_Finalize();
    return result;
}
