/* test_scan.c — unit test for the parallel inclusive prefix-sum kernel.
 *
 * Every MPI rank calls scan() the same number of times (otherwise mpirun
 * would hang).  Only rank 0 checks the result against the oracle in
 * reference.h and prints the outcome.
 *
 * Build:  mpicc -O2 -Wall -Wextra -Werror -std=c11 test_scan.c scan.o -o test_scan -lm
 * Run:    mpirun --oversubscribe -np 4 ./test_scan
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   /* memset */

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long n = REF_N;   /* 1 000 000 */

    /* ---- Allocate and fill input (every rank holds the full array) ---- */
    double *in  = malloc(sizeof(double) * (size_t)n);
    double *out = malloc(sizeof(double) * (size_t)n);
    if (!in || !out) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) in[i] = in_of(i);
    memset(out, 0, sizeof(double) * (size_t)n);

    /* ---- Call the collective kernel (all ranks participate) ---- */
    scan(in, out, n);

    /* ---- Gather the distributed result to rank 0 ---- */
    /* Each rank owns out[lo..hi); we need the full array on rank 0 to
     * compute the global checksum.  Use MPI_Gatherv. */

    /* Build send-count and displacement arrays (same formula as in scan.c) */
    int *counts = malloc(sizeof(int) * (size_t)size);
    int *displs = malloc(sizeof(int) * (size_t)size);
    if (!counts || !displs) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int r = 0; r < size; r++) {
        long lo_r = n *  r      / size;
        long hi_r = n * (r + 1) / size;
        displs[r] = (int)lo_r;
        counts[r] = (int)(hi_r - lo_r);
    }

    /* This rank's own slice */
    long lo = n *  rank      / size;
    long hi = n * (rank + 1) / size;
    int  my_count = (int)(hi - lo);

    /* Gather into a full buffer on rank 0 */
    double *full = NULL;
    if (rank == 0) {
        full = malloc(sizeof(double) * (size_t)n);
        if (!full) {
            fprintf(stderr, "rank 0: allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gatherv(out + lo,          /* sendbuf: this rank's slice          */
                my_count,          /* sendcount                           */
                MPI_DOUBLE,
                full,              /* recvbuf (meaningful only on rank 0) */
                counts,            /* recvcounts                          */
                displs,            /* displacements                       */
                MPI_DOUBLE,
                0,                 /* root                                */
                MPI_COMM_WORLD);

    /* ---- Rank 0: validate against the oracle ---- */
    if (rank == 0) {
        double cs = checksum(full, n);

        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "inclusive scan checksum matches oracle");

        printf("checksum = %.10g  (expected %.10g, tol %g)\n",
               cs, (double)REF_CHECKSUM, (double)REF_TOL);

        int result = TEST_SUMMARY();

        free(full);
        free(counts);
        free(displs);
        free(in);
        free(out);

        MPI_Finalize();
        return result;
    }

    /* Non-rank-0 cleanup */
    free(counts);
    free(displs);
    free(in);
    free(out);

    MPI_Finalize();
    return 0;
}
