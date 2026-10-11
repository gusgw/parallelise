#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cg.h"
#include "reference.h"
#include "problem.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long m = REF_M;
    int iters = REF_ITERS;
    double *x = (double *)malloc(m * m * sizeof(double));
    double *b = (double *)malloc(m * m * sizeof(double));

    for (long i = 0; i < m * m; i++) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    if (rank == 0) {
        double cs = checksum(x, m);
        printf("Checksum: %f\n", cs);
        if (fabs(cs - REF_CHECKSUM) < REF_TOL) {
            printf("PASS\n");
        } else {
            printf("FAIL (diff: %f)\n", fabs(cs - REF_CHECKSUM));
            free(x);
            free(b);
            MPI_Finalize();
            return 1;
        }
    }

    free(x);
    free(b);
    MPI_Finalize();
    return 0;
}
