#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "scan.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    double *in = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    scan(in, out, n);

    /* To compute the global checksum, each rank computes the sum of its block,
     * then we reduce it to rank 0. */
    int r_rank, r_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &r_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &r_size);
    long lo = (long)((double)n * r_rank / r_size);
    long hi = (long)((double)n * (r_rank + 1) / r_size);

    double local_checksum = 0.0;
    for (long i = lo; i < hi; i++) {
        local_checksum += out[i];
    }

    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Checksum: %f (Expected: %f)\n", global_checksum, REF_CHECKSUM);
        if (fabs(global_checksum - REF_CHECKSUM) > REF_TOL) {
            printf("FAILED: Difference %f exceeds tolerance %f\n", 
                   fabs(global_checksum - REF_CHECKSUM), REF_TOL);
            free(in);
            free(out);
            MPI_Finalize();
            return 1;
        }
        printf("PASSED\n");
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
