#include "kernel_b.h"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void kernel_b(double *u, long n, int iters) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double *u_local = malloc((hi - lo) * sizeof(double));
    MPI_Scatter(u, hi - lo, MPI_DOUBLE, u_local, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        double *u_new = malloc((hi - lo) * sizeof(double));

        // Perform computation on local data
        for (long j = lo; j < hi; j++) {
            if (j == 0) {
                u_new[j - lo] = (u_local[j - lo + 1] + u_local[j - lo]) / 2;
            } else if (j == n - 1) {
                u_new[j - lo] = (u_local[j - lo - 1] + u_local[j - lo]) / 2;
            } else {
                u_new[j - lo] = (u_local[j - lo - 1] + u_local[j - lo + 1] + u_local[j - lo]) / 3;
            }
        }

        // Swap u_local and u_new
        double *temp = u_local;
        u_local = u_new;
        u_new = temp;

        free(u_new);
    }

    MPI_Gather(u_local, hi - lo, MPI_DOUBLE, u, hi - lo, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(u_local);
}
