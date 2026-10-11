#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel_b.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Allocate and initialize the grid */
    double *u = malloc(sizeof(double) * REF_N * REF_N);
    if (!u) {
        if (rank == 0) {
            fprintf(stderr, "Memory allocation failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Initialize using u_init from problem.h */
    for (long i = 0; i < REF_N; i++) {
        for (long j = 0; j < REF_N; j++) {
            u[i * REF_N + j] = u_init(i, j, REF_N);
        }
    }

    /* Call the kernel - all ranks must call it */
    kernel_b(u, REF_N, (int)REF_ITERS);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(u, REF_N);
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
