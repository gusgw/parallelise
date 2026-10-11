#include <mpi.h>
#include <stdlib.h>
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
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!in || !out) {
        if (rank == 0) {
            CHECK(0, "allocation failed");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill deterministic input – same on every rank */
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    scan(in, out, n);

    /* Compute checksum of the locally owned portion */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double local_sum = 0.0;
    for (long i = lo; i < hi; ++i) {
        local_sum += out[i];
    }
    double total_checksum = 0.0;
    MPI_Reduce(&local_sum, &total_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CHECK_CLOSE(total_checksum, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(in);
    free(out);

    MPI_Finalize();
    return TEST_SUMMARY();
}
