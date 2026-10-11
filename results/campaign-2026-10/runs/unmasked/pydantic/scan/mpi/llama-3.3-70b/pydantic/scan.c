#include "scan.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void scan(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;

    // Local block scan
    for (long i = lo; i < hi; i++) {
        if (i == lo) {
            out[i] = in[i];
        } else {
            out[i] = out[i - 1] + in[i];
        }
    }

    // Share block totals and compute starting offset
    double block_total;
    if (lo == 0) {
        block_total = 0;
    } else {
        block_total = out[lo - 1];
    }
    double offset;
    MPI_Allgather(&block_total, 1, MPI_DOUBLE, &offset, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    offset = offset * rank;

    // Add offset to local block
    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
}
