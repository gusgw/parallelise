#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_b.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* All ranks allocate and initialize the full grid */
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize grid: only rank 0 needs to do this, but all can do it */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* All ranks call the kernel collectively */
    kernel_b(u, n, iters);

    /* Compute checksum */
    double cs = checksum(u, n);

    /* Only rank 0 checks and reports results */
    if (rank == 0) {
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
