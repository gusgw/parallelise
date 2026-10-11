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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!in || !out) {
        if (rank == 0) fprintf(stderr, "allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* All ranks must call the kernel the same number of times. */
    scan(in, out, n);

    /* Compute local checksum for this rank's block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double local_chk = 0.0;
    for (long i = lo; i < hi; ++i) {
        local_chk += out[i];
    }

    double total_chk = 0.0;
    MPI_Reduce(&local_chk, &total_chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CHECK_CLOSE(total_chk, REF_CHECKSUM, REF_TOL, "checksum");
        if (tests_failed == 0) {
            printf("PASS checksum = %.12g\n", total_chk);
        }
    }

    free(in);
    free(out);
    MPI_Finalize();
    return TEST_SUMMARY();
}
