#include "kernel_b.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void kernel_b(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double *u_local = malloc((hi - lo) * sizeof(double));
    MPI_Scatter(u, hi - lo, MPI_DOUBLE, u_local, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < iters; i++) {
        // Perform computation on local data
        for (long j = lo; j < hi; j++) {
            u_local[j - lo] += 1.0;
        }
        // Combine partial results
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, u_local, hi - lo, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Gather(u_local, hi - lo, MPI_DOUBLE, u, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(u_local);
}
