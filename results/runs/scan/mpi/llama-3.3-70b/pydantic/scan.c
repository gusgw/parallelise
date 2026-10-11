#include "scan.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>

void scan(const double *in, double *out, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long block_size = hi - lo;

    // Local block scan
    double *block_out = (double *)malloc(block_size * sizeof(double));
    double run = 0.0;
    for (long i = lo; i < hi; i++) {
        run += in[i];
        block_out[i - lo] = run;
    }

    // MPI_Exscan of block totals
    double block_total = run;
    double offset;
    MPI_Exscan(&block_total, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        offset = 0.0;
    }

    // Add offset to local block
    for (long i = lo; i < hi; i++) {
        out[i] = block_out[i - lo] + offset;
    }

    free(block_out);
}
