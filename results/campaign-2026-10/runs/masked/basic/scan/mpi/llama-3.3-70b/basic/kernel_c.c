#include "kernel_c.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kernel_c(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++) {
        out[i] = in[i] * in[i];
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, out, n, MPI_DOUBLE, MPI_COMM_WORLD);
}