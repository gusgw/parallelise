#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_b.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = 100;  // More iterations
    
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    kernel_b(u, n, iters);

    double cs = checksum(u, n);
    
    if (rank == 0) {
        printf("Checksum: %.12f\n", cs);
        printf("Expected: %.12f\n", REF_CHECKSUM);
        printf("Diff: %.12f\n", fabs(cs - REF_CHECKSUM));
    }

    free(u);
    MPI_Finalize();
    return 0;
}
