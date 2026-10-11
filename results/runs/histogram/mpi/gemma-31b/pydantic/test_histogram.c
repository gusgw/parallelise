#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "histogram.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *x = (double *)malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        x[i] = x_of(i);
    }

    long *counts = (long *)malloc(REF_NBINS * sizeof(long));

    histogram(x, REF_N, counts, (int)REF_NBINS);

    double cs = checksum(counts, (int)REF_NBINS);

    if (rank == 0) {
        printf("Checksum: %f (Expected: %f, Tol: %f)\n", cs, REF_CHECKSUM, REF_TOL);
        if (fabs(cs - REF_CHECKSUM) > REF_TOL) {
            printf("FAILED\n");
            free(x);
            free(counts);
            MPI_Finalize();
            return 1;
        }
        printf("PASSED\n");
    }

    free(x);
    free(counts);
    MPI_Finalize();
    return 0;
}
