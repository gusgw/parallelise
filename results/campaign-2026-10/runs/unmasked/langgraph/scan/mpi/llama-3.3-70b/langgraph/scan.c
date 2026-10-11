#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scan.h"
#include <mpi.h>

void scan(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double prefix_sum = 0;
    for (long i = lo; i < hi; i++) {
        prefix_sum += in[i];
        out[i] = prefix_sum;
    }
    // All-gather the prefix sums
    double *prefix_sums = malloc(size * sizeof(double));
    MPI_Allgather(&prefix_sum, 1, MPI_DOUBLE, prefix_sums, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    // Calculate the starting offset for each rank
    double offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += prefix_sums[i];
    }
    // Add the offset to the local block
    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
    free(prefix_sums);
}