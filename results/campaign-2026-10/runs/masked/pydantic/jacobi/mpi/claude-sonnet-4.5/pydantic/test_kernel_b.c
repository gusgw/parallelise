#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = REF_ITERS;

    // Allocate and initialize u
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    // Run kernel
    kernel_b(u, n, iters);

    // Only rank 0 checks and reports
    int result = 0;
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        result = TEST_SUMMARY();
    }

    free(u);
    MPI_Finalize();
    return result;
}
