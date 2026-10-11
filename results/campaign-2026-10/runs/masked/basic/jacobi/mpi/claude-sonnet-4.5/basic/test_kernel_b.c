#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = REF_N;
    int iters = REF_ITERS;

    // Allocate and initialize grid
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

    // Check result on rank 0
    if (rank == 0) {
        double result = checksum(u, n);
        double error = fabs(result - REF_CHECKSUM);
        
        if (error < REF_TOL) {
            printf("PASS: checksum=%.10f (expected %.10f, error %.10f)\n",
                   result, REF_CHECKSUM, error);
        } else {
            printf("FAIL: checksum=%.10f (expected %.10f, error %.10f)\n",
                   result, REF_CHECKSUM, error);
            free(u);
            MPI_Finalize();
            return 1;
        }
    }

    free(u);
    MPI_Finalize();
    return 0;
}
