#include "scan.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));
    for (long i = 0; i < REF_N; i++) {
        in[i] = ((i * 2654435761) % 1000003) / 1000003.0 - 0.5;
    }
    scan(in, out, REF_N);
    if (rank == 0) {
        double checksum = 0.0;
        for (long i = 0; i < REF_N; i++) {
            checksum += out[i];
        }
        assert(fabs(checksum - REF_CHECKSUM) <= REF_TOL);
        printf("checksum = %f\n", checksum);
    }
    MPI_Finalize();
    return 0;
}
