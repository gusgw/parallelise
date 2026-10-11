#include <mpi.h>
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

    long n = REF_N;
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!in || !out) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill deterministic input */
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    /* Run the kernel */
    scan(in, out, n);

    /* Determine this rank's block */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    /* Compute local checksum of the block we own */
    double local_cs = 0.0;
    for (long i = lo; i < hi; ++i) {
        local_cs += out[i];
    }

    double global_cs = 0.0;
    MPI_Reduce(&local_cs, &global_cs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CHECK_CLOSE(global_cs, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(in);
    free(out);

    MPI_Finalize();
    return TEST_SUMMARY();
}
