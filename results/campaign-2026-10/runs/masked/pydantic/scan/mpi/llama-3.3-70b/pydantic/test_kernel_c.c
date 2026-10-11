#include "kernel_c.h"
#include "reference.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int main() {
    int rank;
    MPI_Init(NULL, NULL);
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
    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}
