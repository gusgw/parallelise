#include "scan.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));

    // Fill in with deterministic values
    for (long i = 0; i < REF_N; i++) {
        in[i] = ((i * 2654435761) % 1000003) / 1000003.0 - 0.5;
    }

    // Call the kernel the same number of times on every rank
    for (int i = 0; i < 10; i++) {
        scan(in, out, REF_N);
    }

    double checksum = 0.0;
    for (long i = 0; i < REF_N; i++) {
        checksum += out[i];
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        assert(fabs(checksum - REF_CHECKSUM) <= REF_TOL);
        printf("Checksum matches reference: %.6f\n", checksum);
    }

    MPI_Finalize();
    return 0;
}
