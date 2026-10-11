#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "integrate.h"

/* Parallel-performance harness. Sweeps thread counts and prints one CSV row per
 * count:  threads,time_s,speedup,efficiency,msamples_per_s
 * The number of subintervals n is argv[1] (default 1e9); throughput counts n
 * samples per run. Time is wall time from omp_get_wtime; speedup and efficiency
 * are relative to the single-thread run. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 1000000000L;

    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) {
        sweep[ns++] = t;
    }
    if (ns == 0 || sweep[ns - 1] != maxt) {
        sweep[ns++] = maxt; /* always include the full core count */
    }

    printf("threads,time_s,speedup,efficiency,msamples_per_s\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        int t = sweep[i];
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        volatile double r = integrate(0.0, 1.0, n);
        double elapsed = omp_get_wtime() - start;
        (void)r;
        if (i == 0) {
            t1 = elapsed;
        }
        double speedup = t1 / elapsed;
        double efficiency = speedup / (double)t;
        double msamples = (double)n / elapsed / 1.0e6;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", t, elapsed, speedup, efficiency, msamples);
    }
    return 0;
}
