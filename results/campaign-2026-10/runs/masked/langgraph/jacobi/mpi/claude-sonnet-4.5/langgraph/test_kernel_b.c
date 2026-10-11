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
    
    // Allocate and initialize
    double *u = malloc(sizeof(double) * n * n);
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    // Run kernel (collective call)
    kernel_b(u, n, iters);
    
    // Only rank 0 checks and reports
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        int result = TEST_SUMMARY();
        free(u);
        MPI_Finalize();
        return result;
    }
    
    free(u);
    MPI_Finalize();
    return 0;
}
