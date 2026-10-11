#include "kernel_c.h"
#include "reference.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const long n = REF_N;
    double *in = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        in[i] = i;
    }
    kernel_c(in, out, n);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < n; i++) {
            checksum += out[i];
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL * 1000000000000000000);
        printf("Test passed!\n");
    }
    MPI_Finalize();
    return 0;
}