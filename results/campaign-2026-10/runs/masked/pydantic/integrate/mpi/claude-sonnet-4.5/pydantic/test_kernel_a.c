#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* All ranks must call the kernel */
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);
    
    int ret = 0;
    /* Only rank 0 checks and reports */
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        ret = TEST_SUMMARY();
    }
    
    MPI_Finalize();
    return ret;
}
