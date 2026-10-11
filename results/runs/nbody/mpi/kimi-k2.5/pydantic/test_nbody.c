#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "nbody.h"
#include "problem.h"
#include "reference.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;

    /* Allocate and fill position arrays (all ranks have full copy) */
    double *px = (double *)malloc(n * sizeof(double));
    double *py = (double *)malloc(n * sizeof(double));
    double *ax = (double *)malloc(n * sizeof(double));
    double *ay = (double *)malloc(n * sizeof(double));

    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    /* Call the collective kernel (all ranks call together) */
    nbody_accel(px, py, ax, ay, n);

    /* Compute checksum and verify (only rank 0 checks) */
    if (rank == 0) {
        double cs = checksum(ax, ay, n);
        double err = fabs(cs - REF_CHECKSUM);
        int pass = err <= REF_TOL;

        printf("checksum = %.16e\n", cs);
        printf("expected = %.16e\n", REF_CHECKSUM);
        printf("error    = %.16e\n", err);
        printf("tol      = %.16e\n", REF_TOL);
        printf("%s\n", pass ? "PASS" : "FAIL");

        if (!pass) {
            MPI_Finalize();
            return 1;
        }
    }

    free(px);
    free(py);
    free(ax);
    free(ay);

    MPI_Finalize();
    return 0;
}
