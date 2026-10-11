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

    /* Allocate and fill input array (same on all ranks) */
    double *in = (double *)malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Allocate output array */
    double *out = (double *)malloc(n * sizeof(double));

    /* Call the scan kernel (collective) */
    scan(in, out, n);

    /* Compute this rank's block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Gather all results to rank 0 for checking */
    double *full_out = NULL;
    if (rank == 0) {
        full_out = (double *)malloc(n * sizeof(double));
    }

    /* Use MPI_Gatherv to collect all blocks */
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            long r_lo = n * r / size;
            long r_hi = n * (r + 1) / size;
            recvcounts[r] = (int)(r_hi - r_lo);
            displs[r] = (int)r_lo;
        }
    }

    MPI_Gatherv(&out[lo], (int)(hi - lo), MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(full_out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "scan checksum");
        
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
