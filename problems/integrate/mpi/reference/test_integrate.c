#include <math.h>
#include <mpi.h>

#include "integrate.h"
#include "reference.h"
#include "testing.h"

/* integrate() is collective, so every rank must call it the same number of times;
 * only rank 0 asserts on the (identical, Allreduced) result and reports. Run with
 *   mpirun -np <ranks> ./test_integrate
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double got = integrate(REF_A, REF_B, REF_N_TEST);
    double coarse = integrate(REF_A, REF_B, 1000L);
    double fine = integrate(REF_A, REF_B, 100000L);

    int rc = 0;
    if (rank == 0) {
        CHECK_CLOSE(got, REF_EXPECTED, REF_TOL, "integrate([0,1]) approximates pi");
        CHECK_CLOSE(coarse, REF_EXPECTED, 1e-3, "coarse n is still near pi");
        CHECK(fabs(fine - REF_EXPECTED) < fabs(coarse - REF_EXPECTED),
              "finer n is more accurate than coarser n");
        rc = TEST_SUMMARY();
    }
    MPI_Finalize();
    return rc;
}
