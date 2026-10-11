#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "midpoint.h"

/* Sweeps thread counts and prints threads,time_s,speedup,efficiency,msamples_per_s. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 20000000L;
    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) {
        sweep[ns++] = t;
    }
    if (sweep[ns - 1] != maxt) {
        sweep[ns++] = maxt;
    }
    printf("threads,time_s,speedup,efficiency,msamples_per_s\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        omp_set_num_threads(sweep[i]);
        double start = omp_get_wtime();
        volatile double r = midpoint(0.0, 3.141592653589793, n);
        double elapsed = omp_get_wtime() - start;
        (void)r;
        if (i == 0) {
            t1 = elapsed;
        }
        double speedup = t1 / elapsed;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", sweep[i], elapsed, speedup,
               speedup / sweep[i], (double)n / elapsed / 1.0e6);
    }
    return 0;
}
