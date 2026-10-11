#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "reference.h"
#include "scan.h"
#include <mpi.h>

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = REF_N * rank / size;
    long hi = REF_N * (rank + 1) / size;
    double *in = malloc(REF_N * sizeof(double));
    double *out = malloc(REF_N * sizeof(double));
    for (long i = lo; i < hi; i++) {
        in[i] = i;
    }
    scan(in, out, REF_N);
    double checksum = 0;
    for (long i = lo; i < hi; i++) {
        checksum += out[i];
    }
    if (rank == 0) {
        double expected_checksum = REF_CHECKSUM;
        if (fabs(checksum - expected_checksum) > REF_TOL) {
            printf("Error: checksum does not match\n");
            return 1;
        }
    }
    MPI_Finalize();
    return 0;
}