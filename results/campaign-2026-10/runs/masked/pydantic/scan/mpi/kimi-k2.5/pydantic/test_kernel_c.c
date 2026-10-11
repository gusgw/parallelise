#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_c.h"
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
        if (rank == 0) fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill input using problem.h helper */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Call the kernel - all ranks must call it */
    kernel_c(in, out, n);

    /* Gather all results to rank 0 for verification */
    double *gathered_out = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        gathered_out = malloc(sizeof(double) * n);
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            long l = n * i / size;
            long h = n * (i + 1) / size;
            recvcounts[i] = (int)(h - l);
            displs[i] = (int)l;
        }
    }

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    MPI_Gatherv(out + lo, (int)local_n, MPI_DOUBLE,
                gathered_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(gathered_out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

        int failed = TEST_SUMMARY();
        free(gathered_out);
        free(recvcounts);
        free(displs);
        free(in);
        free(out);
        MPI_Finalize();
        return failed;
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
