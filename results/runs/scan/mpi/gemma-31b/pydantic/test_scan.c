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

    double *in = malloc(REF_N * sizeof(double));
    double *out = malloc(REF_N * sizeof(double));

    for (long i = 0; i < REF_N; i++) {
        in[i] = in_of(i);
    }

    scan(in, out, REF_N);

    long lo = (long)((double)REF_N * rank / size);
    long hi = (long)((double)REF_N * (rank + 1) / size);

    double local_checksum = 0.0;
    for (long i = lo; i < hi; i++) {
        local_checksum += out[i];
    }

    double total_checksum = 0.0;
    MPI_Reduce(&local_checksum, &total_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double diff = fabs(total_checksum - REF_CHECKSUM);
        printf("Checksum: %f, Expected: %f, Diff: %f\n", total_checksum, REF_CHECKSUM, diff);
        if (diff <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            free(in);
            free(out);
            MPI_Finalize();
            return 1;
        }
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
