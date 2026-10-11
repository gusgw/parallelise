#include "kernel_c.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

void kernel_c(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double *partial_out = (double *)malloc((hi - lo) * sizeof(double));
    for (long i = lo; i < hi; i++) {
        partial_out[i - lo] = in[i] * in[i];
    }
    MPI_Allgather(partial_out, hi - lo, MPI_DOUBLE, out, hi - lo, MPI_DOUBLE, MPI_COMM_WORLD);
    free(partial_out);
}
