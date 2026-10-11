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
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill input using the provided generator.
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    // Call the kernel (each rank writes only its slice of out).
    kernel_c(in, out, n);

    // Gather full result onto rank 0 for checksum.
    double *full_out = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        full_out = (double *)malloc(n * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        if (!full_out || !recvcounts || !displs) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Compute local slice indices.
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    // Prepare counts/displacements on root.
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            long r_lo = n * r / size;
            long r_hi = n * (r + 1) / size;
            recvcounts[r] = (int)(r_hi - r_lo);
            displs[r] = (int)r_lo;
        }
    }

    MPI_Gatherv(out + lo, (int)local_n, MPI_DOUBLE,
                full_out, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double chk = 0.0;
    if (rank == 0) {
        chk = checksum(full_out, n);
    }

    if (rank == 0) {
        CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    }

    free(in);
    free(out);
    if (rank == 0) {
        free(full_out);
        free(recvcounts);
        free(displs);
    }
    MPI_Finalize();
    return TEST_SUMMARY();
}
