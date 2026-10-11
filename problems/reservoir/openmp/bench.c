#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "reservoir.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). Sweeps thread counts and prints one
 * CSV row per count:  threads,time_s,speedup,efficiency,melems_per_s
 * Array length n is argv[1] (default 134217728). Speedup and efficiency are relative to
 * the single-thread run, so a kernel that is not really parallel shows no speedup. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 134217728L;
    double *a = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!a || !out) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < n; i++) a[i] = change_of(i);

    /* one untimed warm-up run so the first-write page faults on out are paid before the
     * timed sweep begins; otherwise the single-thread baseline is charged for them and a
     * purely serial kernel shows a false speed-up on the later (warm) runs. */
    reservoir(a, out, n);

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
        reservoir(a, out, n);
        double elapsed = omp_get_wtime() - start;
        if (i == 0) t1 = elapsed;
        double speedup = t1 / elapsed;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", t, elapsed, speedup, speedup / (double)t,
               (double)n / elapsed / 1.0e6);
    }
    free(a); free(out);
    return 0;
}
