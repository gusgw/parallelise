#include "scan.h"
#include "reference.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);

    double *in = (double *)malloc(REF_N * sizeof(double));
    double *out = (double *)malloc(REF_N * sizeof(double));

    // Initialize input
    for (long i = 0; i < REF_N; i++) {
        in[i] = i;
    }

    // Call scan kernel
    scan(in, out, REF_N);

    // Check result
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < REF_N; i++) {
            checksum += out[i];
        }
        if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
            printf("Error: checksum mismatch\n");
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
