#include "histogram.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);
    double *x = (double *)malloc(REF_N * sizeof(double));
    long *counts = (long *)malloc(REF_NBINS * sizeof(long));
    for (long i = 0; i < REF_N; i++) {
        x[i] = ((i * 2654435761 + 12345) % 4294967296) / 4294967296.0;
    }
    histogram(x, REF_N, counts, REF_NBINS);
    double checksum = 0.0;
    for (int b = 0; b < REF_NBINS; b++) {
        checksum += counts[b] * (b + 1);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        assert(fabs(checksum - REF_CHECKSUM) <= REF_TOL);
        printf("PASS\n");
    }
    MPI_Finalize();
    free(x);
    free(counts);
    return 0;
}
