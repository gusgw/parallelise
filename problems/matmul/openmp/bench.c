#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). Sweeps thread counts and prints one
 * CSV row per count:  threads,time_s,speedup,efficiency,gflops
 * Matrix size n is argv[1] (default 1024). Speedup and efficiency are relative to the
 * single-thread run, so a kernel that is not really parallel shows no speedup. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) sweep[ns++] = t;
    if (ns == 0 || sweep[ns - 1] != maxt) sweep[ns++] = maxt;

    printf("threads,time_s,speedup,efficiency,gflops\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        int t = sweep[i];
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        matmul(A, B, C, n);
        double elapsed = omp_get_wtime() - start;
        if (i == 0) t1 = elapsed;
        double speedup = t1 / elapsed;
        printf("%d,%.6f,%.3f,%.3f,%.2f\n", t, elapsed, speedup, speedup / (double)t,
               2.0 * (double)n * (double)n * (double)n / elapsed / 1.0e9);
    }
    free(A); free(B); free(C);
    return 0;
}
