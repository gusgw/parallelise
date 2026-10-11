#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* Initialize grid on all ranks */
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(n * n * sizeof(double));
    
    /* Initialize with u_init */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    /* Collective call: all ranks participate */
    jacobi(u, n, iters);
    
    /* Only rank 0 checks the result */
    if (rank == 0) {
        double cs = checksum(u, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");
        printf("checksum = %.12f\n", cs);
    }
    
    free(u);
    
    MPI_Finalize();
    
    /* Return test result from rank 0 */
    if (rank == 0) {
        return TEST_SUMMARY();
    }
    return 0;
}
