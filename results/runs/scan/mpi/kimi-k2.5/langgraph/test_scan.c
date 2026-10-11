#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "scan.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Allocate full arrays (each rank holds a full copy of input) */
    double *in = malloc(REF_N * sizeof(double));
    double *out = malloc(REF_N * sizeof(double));

    /* Fill input deterministically */
    for (long i = 0; i < REF_N; i++) {
        in[i] = in_of(i);
    }

    /* Call the collective kernel */
    scan(in, out, REF_N);

    /* Compute this rank's block bounds */
    long lo = REF_N * rank / size;
    long hi = REF_N * (rank + 1) / size;

    /* Compute local checksum from this rank's valid output block */
    double local_cs = 0.0;
    for (long i = lo; i < hi; i++) {
        local_cs += out[i];
    }

    /* Reduce to get global checksum on rank 0 */
    double cs = 0.0;
    MPI_Reduce(&local_cs, &cs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Only rank 0 validates and prints */
    if (rank == 0) {
        double err = fabs(cs - REF_CHECKSUM);
        printf("checksum = %.16f\n", cs);
        printf("expected = %.16f\n", REF_CHECKSUM);
        printf("error    = %.16e\n", err);
        printf("tol      = %.16e\n", REF_TOL);

        if (err > REF_TOL) {
            printf("FAIL: error exceeds tolerance\n");
            free(in);
            free(out);
            MPI_Finalize();
            return 1;
        }
        printf("PASS\n");
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
