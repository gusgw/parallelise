#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "cg.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). Sweeps thread counts and prints one
 * CSV row per count:  threads,time_s,speedup,efficiency,mcells_per_s
 * Grid side m is argv[1] (default 1024), CG iterations argv[2] (default 100). Each
 * measurement solves the same fixed-iteration system from scratch (cgsolve starts from
 * x = 0, so no re-fill is needed). Speedup and efficiency are relative to the
 * single-thread run, so a kernel that is not really parallel shows no speedup.
 * mcells_per_s counts grid-cell updates per second: m*m*iters/time/1e6. */
int main(int argc, char **argv)
{
    long m = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    if (!x || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long t = 0; t < m * m; t++) b[t] = b_of(t);

    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) sweep[ns++] = t;
    if (ns == 0 || sweep[ns - 1] != maxt) sweep[ns++] = maxt;

    printf("threads,time_s,speedup,efficiency,mcells_per_s\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        int t = sweep[i];
        omp_set_num_threads(t);
        double start = omp_get_wtime();
        cgsolve(x, b, m, iters);
        double elapsed = omp_get_wtime() - start;
        if (i == 0) t1 = elapsed;
        double speedup = t1 / elapsed;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", t, elapsed, speedup, speedup / (double)t,
               (double)m * (double)m * (double)iters / elapsed / 1.0e6);
    }
    free(x); free(b);
    return 0;
}
