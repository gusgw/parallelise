#include <mpi.h>
#include <stdlib.h>

#include "kernel_c.h"
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
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!in || !out) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill input array
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    // Run kernel
    kernel_c(in, out, n);

    // Determine local range (same partition as kernel)
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    // Compute local checksum of the portion we own
    double local_cs = 0.0;
    for (long i = lo; i < hi; ++i) {
        local_cs += out[i];
    }

    // Reduce to global checksum on rank 0
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
