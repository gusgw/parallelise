#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = (int)REF_ITERS;

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

    if (rank == 0) {
        printf("Initial checksum: %f\n", checksum(u, n));
    }

    jacobi(u, n, iters);

    if (rank == 0) {
        double cs = checksum(u, n);
        printf("Final checksum: %.12f\n", cs);
        printf("Expected: %.12f\n", REF_CHECKSUM);
        
        // Print row sums
        for (long i = 0; i < n; i++) {
            double row_sum = 0;
            for (long j = 0; j < n; j++) {
                row_sum += u[i * n + j];
            }
            if (row_sum > 0.001) {
                printf("Row %ld sum: %f\n", i, row_sum);
            }
        }
    }

    free(u);
    MPI_Finalize();
    return 0;
}
