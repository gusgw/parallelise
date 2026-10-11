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
    double *block_in = (double *)malloc(block_size * sizeof(double));
    double *block_out = (double *)malloc(block_size * sizeof(double));
    MPI_Scatter(in, block_size, MPI_DOUBLE, block_in, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double block_sum = 0.0;
    for (long i = 0; i < block_size; i++) {
        block_sum += block_in[i];
        block_out[i] = block_sum;
    }
    double offset = 0.0;
    if (rank > 0) {
        MPI_Recv(&offset, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (long i = 0; i < block_size; i++) {
        block_out[i] += offset;
    }
    if (rank < size - 1) {
        double send_offset = offset + block_sum;
        MPI_Send(&send_offset, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
    MPI_Gather(block_out, block_size, MPI_DOUBLE, out, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(block_in);
    free(block_out);
}
