#include <mpi.h>
#include <stdlib.h>

#include "reservoir.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). reservoir() is collective and leaves the
 * result DISTRIBUTED (each rank's own block of out is correct); the TEST gathers the
 * blocks to rank 0 -- outside the kernel, as the contract requires -- and rank 0 checks
 * the full checksum against the oracle. */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    double *a = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) a[i] = change_of(i);

    reservoir(a, out, n);

    /* gather the per-rank blocks of out to rank 0 for checking */
    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int r = 0; r < size; r++) {
        long rlo = n * r / size, rhi = n * (r + 1) / size;
        counts[r] = (int)(rhi - rlo);
        displs[r] = (int)rlo;
    }
    const long lo = n * rank / size;
    double *full = NULL;
    if (rank == 0)
        full = malloc(sizeof(double) * n);
    MPI_Gatherv(&out[lo], counts[rank], MPI_DOUBLE,
                full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        double cs = checksum(full, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "reservoir checksum matches the oracle");
        rc = TEST_SUMMARY();
        free(full);
    }
    free(a); free(out); free(counts); free(displs);
    MPI_Finalize();
    return rc;
}
