#include "histogram.h"
#include "problem.h"
#include "reference.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    double *x = (double *)malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) x[i] = x_of(i);

    long *counts = (long *)malloc(REF_NBINS * sizeof(long));
    histogram(x, REF_N, counts, REF_NBINS);

    double checksum_value = checksum(counts, REF_NBINS);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Checksum: %f\n", checksum_value);
        if (fabs(checksum_value - REF_CHECKSUM) <= REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
        }
    }

    MPI_Finalize();
    return 0;
}
