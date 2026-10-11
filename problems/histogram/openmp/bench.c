#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "histogram.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). Sweeps thread counts and prints one
 * CSV row per count:  threads,time_s,speedup,efficiency,melems_per_s
 * Value count n is argv[1] (default 200000000 -- a 1.6 GB input array), bin count
 * argv[2] (default 256). Speedup and efficiency are relative to the single-thread
 * run, so a kernel that is not really parallel shows no speedup. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 200000000L;
    int nbins = (argc > 2) ? atoi(argv[2]) : 256;
    double *x = malloc(sizeof(double) * n);
    long *counts = malloc(sizeof(long) * nbins);
    if (!x || !counts) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < n; i++) x[i] = x_of(i);

    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) sweep[ns++] = t;
    if (ns == 0 || sweep[ns - 1] != maxt) sweep[ns++] = maxt;

    printf("threads,time_s,speedup,efficiency,melems_per_s\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        int t = sweep[i];
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        histogram(x, n, counts, nbins);
        double elapsed = omp_get_wtime() - start;
        if (i == 0) t1 = elapsed;
        double speedup = t1 / elapsed;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", t, elapsed, speedup, speedup / (double)t,
               (double)n / elapsed / 1.0e6);
    }
    free(x); free(counts);
    return 0;
}
