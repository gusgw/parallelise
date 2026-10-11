#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "scan.h"
#include "reference.h"
#include "problem.h"
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

    /* Only rank 0 verifies the result */
    if (rank == 0) {
        /* Gather the full output from all ranks to verify checksum */
        double *full_out = malloc(sizeof(double) * n);
        if (!full_out) {
            fprintf(stderr, "alloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Copy rank 0's portion */
        long lo = n * rank / size;
        long hi = n * (rank + 1) / size;
        memcpy(full_out + lo, out + lo, sizeof(double) * (hi - lo));

        /* Receive from other ranks */
        for (int src = 1; src < size; src++) {
            long src_lo = n * src / size;
            long src_hi = n * (src + 1) / size;
            MPI_Recv(full_out + src_lo, (int)(src_hi - src_lo), MPI_DOUBLE,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Verify checksum */
        double cs = checksum(full_out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

        free(full_out);

        int result = TEST_SUMMARY();
        free(in);
        free(out);
        MPI_Finalize();
        return result;
    } else {
        /* Send local portion to rank 0 */
        long lo = n * rank / size;
        long hi = n * (rank + 1) / size;
        MPI_Send(out + lo, (int)(hi - lo), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        free(in);
        free(out);
        MPI_Finalize();
        return 0;
    }
}
