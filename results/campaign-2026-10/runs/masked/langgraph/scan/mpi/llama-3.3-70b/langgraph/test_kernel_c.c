#include "kernel_c.h"
#include "reference.h"
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const long n = REF_N;
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        in[i] = i;
    }
    kernel_c(in, out, n);
    if (rank == 0) {
        double checksum = 0;
        for (long i = 0; i < n; i++) {
            checksum += out[i];
        }
        assert(fabs(checksum - REF_CHECKSUM) < REF_TOL);
        printf("Test passed!\n");
    }
    MPI_Finalize();
    free(in);
    free(out);
    return 0;
}
