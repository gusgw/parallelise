#include "kernel_c.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>

void kernel_c(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double local_sum = 0;
    for (long i = lo; i < hi; i++) {
        local_sum += in[i];
    }
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (long i = lo; i < hi; i++) {
        out[i] = global_sum;
    }
}
