#include "histogram.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *x = (double *)malloc(REF_N * sizeof(double));
    long *counts = (long *)malloc(REF_NBINS * sizeof(long));
    for (long i = 0; i < REF_N; i++) {
        x[i] = ((i * 2654435761 + 12345) % 4294967296) / 4294967296.0;
    }
    for (int i = 0; i < REF_NBINS; i++) {
        counts[i] = 0;
    }
    histogram(x, REF_N, counts, REF_NBINS);
    if (rank == 0) {
        double checksum = 0.0;
        for (int b = 0; b < REF_NBINS; b++) {
            checksum += counts[b] * (b + 1);
        }
        assert(fabs(checksum - REF_CHECKSUM) <= REF_TOL);
        printf("PASS: checksum = %f\n", checksum);
    }
    MPI_Finalize();
    free(x);
    free(counts);
    return 0;
}
