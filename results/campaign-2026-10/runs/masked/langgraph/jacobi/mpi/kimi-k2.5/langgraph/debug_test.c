#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "problem.h"
#include "reference.h"

/* Serial version for comparison */
void serial_jacobi(double *u, long n, int iters)
{
    double *un = malloc(sizeof(double) * n * n);
    
    for (int iter = 0; iter < iters; iter++) {
        memcpy(un, u, sizeof(double) * n * n);
        
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (
                    u[(i - 1) * n + j] +
                    u[(i + 1) * n + j] +
                    u[i * n + (j - 1)] +
                    u[i * n + (j + 1)]
                );
            }
        }
        
        memcpy(u, un, sizeof(double) * n * n);
    }
    
    free(un);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = REF_N;
    int iters = 10;  // 10 iterations for debugging

    double *u_serial = malloc(sizeof(double) * n * n);
    double *u_parallel = malloc(sizeof(double) * n * n);
    
    if (!u_serial || !u_parallel) {
        fprintf(stderr, "alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize grids */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u_serial[i * n + j] = u_init(i, j, n);
            u_parallel[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run serial version */
    serial_jacobi(u_serial, n, iters);
    
    /* Run parallel version */
    extern void kernel_b(double *u, long n, int iters);
    kernel_b(u_parallel, n, iters);

    /* Compare results */
    if (rank == 0) {
        int errors = 0;
        double max_diff = 0.0;
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                double diff = fabs(u_serial[i * n + j] - u_parallel[i * n + j]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-10) {
                    if (errors < 10) {
                        printf("Mismatch at (%ld, %ld): serial=%.12f, parallel=%.12f, diff=%.12e\n",
                               i, j, u_serial[i * n + j], u_parallel[i * n + j], diff);
                    }
                    errors++;
                }
            }
        }
        printf("Total errors: %d, max diff: %.12e\n", errors, max_diff);
        
        double cs_serial = checksum(u_serial, n);
        double cs_parallel = checksum(u_parallel, n);
        printf("Serial checksum: %.12f\n", cs_serial);
        printf("Parallel checksum: %.12f\n", cs_parallel);
        printf("Checksum diff: %.12e\n", fabs(cs_serial - cs_parallel));
    }

    free(u_serial);
    free(u_parallel);
    
    MPI_Finalize();
    return 0;
}
