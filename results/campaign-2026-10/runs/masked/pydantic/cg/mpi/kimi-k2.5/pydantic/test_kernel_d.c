#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_d.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    double *x = malloc(sizeof(double) * n);
    double *b = malloc(sizeof(double) * n);
    
    if (!x || !b) {
        if (rank == 0) {
            fprintf(stderr, "Allocation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill b using problem.h helper */
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }

    /* Call the kernel - all ranks must call it */
    cgsolve(x, b, m, iters);

    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(x, m);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
        
        int result = TEST_SUMMARY();
        free(x);
        free(b);
        MPI_Finalize();
        return result;
    }

    free(x);
    free(b);
    MPI_Finalize();
    return 0;
}
