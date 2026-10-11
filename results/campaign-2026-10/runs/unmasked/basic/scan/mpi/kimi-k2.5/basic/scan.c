#include "scan.h"

#include <mpi.h>

void scan(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute local block bounds */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* First pass: compute local sum */
    const double *in_local = in + lo;
    double local_sum = 0.0;
    for (long i = 0; i < local_n; i++) {
        local_sum += in_local[i];
    }

    /* Get offset from previous ranks */
    double offset = 0.0;
    if (size > 1) {
        MPI_Exscan(&local_sum, &offset, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    /* Second pass: compute local scan with offset */
    double *out_local = out + lo;
    double running = offset;
    for (long i = 0; i < local_n; i++) {
        running += in_local[i];
        out_local[i] = running;
    }
}
