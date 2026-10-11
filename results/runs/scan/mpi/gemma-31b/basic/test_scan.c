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
    long len = hi - lo;

    double *full_out = NULL;
    int *recv_counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        full_out = malloc(REF_N * sizeof(double));
        recv_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            long l = (long)((double)REF_N * i / size);
            long h = (long)((double)REF_N * (i + 1) / size);
            recv_counts[i] = (int)(h - l);
            displs[i] = (int)l;
        }
    }

    MPI_Gatherv(&out[lo], (int)len, MPI_DOUBLE, full_out, recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double actual_checksum = checksum(full_out, REF_N);
        printf("Actual checksum: %f\n", actual_checksum);
        printf("Reference checksum: %f\n", REF_CHECKSUM);
        if (fabs(actual_checksum - REF_CHECKSUM) < REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            free(recv_counts);
            free(displs);
            free(full_out);
            free(in);
            free(out);
            MPI_Finalize();
            return 1;
        }
        free(recv_counts);
        free(displs);
        free(full_out);
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
