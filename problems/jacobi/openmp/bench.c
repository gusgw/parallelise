#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). Sweeps thread counts and prints one
 * CSV row per count:  threads,time_s,speedup,efficiency,mcups
 * Grid size n is argv[1] (default 1024), sweeps argv[2] (default 120; more sweeps give
 * steadier timing -- the stencil is memory-bandwidth-bound). Each measurement starts
 * from a fresh initial grid. Speedup is relative to the single-thread run. */
int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 120;
    double *u = malloc(sizeof(double) * n * n);
    if (!u) { fprintf(stderr, "alloc failed\n"); return 1; }

    int maxt = omp_get_max_threads();
    int sweep[32];
    int ns = 0;
    for (int t = 1; t <= maxt; t *= 2) sweep[ns++] = t;
    if (ns == 0 || sweep[ns - 1] != maxt) sweep[ns++] = maxt;

    printf("threads,time_s,speedup,efficiency,mcups\n");
    double t1 = 0.0;
    for (int i = 0; i < ns; i++) {
        int t = sweep[i];
        omp_set_num_threads(t);
        for (long r = 0; r < n; r++)
            for (long c = 0; c < n; c++) u[r * n + c] = u_init(r, c, n);
        double start = omp_get_wtime();
        jacobi(u, n, iters);
        double elapsed = omp_get_wtime() - start;
        if (i == 0) t1 = elapsed;
        double speedup = t1 / elapsed;
        double updates = (double)(n - 2) * (double)(n - 2) * (double)iters;
        printf("%d,%.6f,%.3f,%.3f,%.1f\n", t, elapsed, speedup, speedup / (double)t,
               updates / elapsed / 1.0e6);
    }
    free(u);
    return 0;
}
