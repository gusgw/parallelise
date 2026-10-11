#include "scan.h"
#include "reference.h"
#include "mpi.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* in = (double*) malloc(1000000 * sizeof(double));
    double* out = (double*) malloc(1000000 * sizeof(double));

    for (long i = 0; i < 1000000; i++) {
        in[i] = ((i * 2654435761) % 1000003) / 1000003.0 - 0.5;
    }

    scan(in, out, 1000000);

    double checksum = 0.0;
    for (long i = 0; i < 1000000; i++) {
        checksum += out[i];
    }

    if (rank == 0) {
        assert(fabs(checksum - 3178464.2888227087) < 1e-4);
        printf("checksum = %f\n", checksum);
    }

    MPI_Finalize();
    return 0;
}
