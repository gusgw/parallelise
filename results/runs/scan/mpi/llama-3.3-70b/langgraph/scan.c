#include "scan.h"
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

void scan(const double* in, double* out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    double* local_out = (double*) malloc((hi - lo) * sizeof(double));
    double run = 0.0;
    for (long i = lo; i < hi; i++) {
        run += in[i];
        local_out[i - lo] = run;
    }

    double* global_out = (double*) malloc(n * sizeof(double));
    MPI_Allgather(local_out, hi - lo, MPI_DOUBLE, global_out, hi - lo, MPI_DOUBLE, MPI_COMM_WORLD);

    double prefix_sum = 0.0;
    for (long i = 0; i < lo; i++) {
        prefix_sum += in[i];
    }
    for (long i = lo; i < hi; i++) {
        out[i] = prefix_sum + local_out[i - lo];
    }
    for (long i = hi; i < n; i++) {
        out[i] = prefix_sum + global_out[i - lo];
    }

    free(local_out);
    free(global_out);
}
