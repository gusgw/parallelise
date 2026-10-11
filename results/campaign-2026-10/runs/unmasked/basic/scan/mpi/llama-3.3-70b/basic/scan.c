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

    // Local scan
    for (long i = lo; i < hi; i++) {
        if (i == lo) {
            out[i] = in[i];
        } else {
            out[i] = out[i - 1] + in[i];
        }
    }

    // Share block totals
    double total;
    if (lo == 0) {
        total = out[hi - 1];
    } else {
        total = out[hi - 1] - out[lo - 1];
    }
    double totals[size];
    MPI_Allgather(&total, 1, MPI_DOUBLE, totals, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    // Calculate offset
    double offset = 0;
    for (int i = 0; i < rank; i++) {
        offset += totals[i];
    }

    // Add offset to local block
    for (long i = lo; i < hi; i++) {
        out[i] += offset;
    }
}
