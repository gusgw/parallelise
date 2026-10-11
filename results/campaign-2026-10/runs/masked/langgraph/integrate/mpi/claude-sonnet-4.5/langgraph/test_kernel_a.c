#include <mpi.h>
#include <stdio.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* All ranks must call the kernel */
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);
    
    /* Only rank 0 checks and reports */
    int test_result = 0;
    if (rank == 0) {
        CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result");
        test_result = TEST_SUMMARY();
    }
    
    MPI_Finalize();
    return test_result;
}
