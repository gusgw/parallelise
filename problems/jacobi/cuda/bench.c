#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jacobi.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). CUDA has no worker-count sweep, so the
 * scaling evidence is serial CPU vs GPU on the same problem (n = argv[1] default 1024,
 * iters = argv[2] default 120):
 *     impl,time_s,speedup   with rows  serial,<t>,1.0  and  gpu,<t>,<t_serial/t_gpu>
 * The serial baseline lives HERE, in trusted code; each measurement starts from a fresh
 * initial grid; the GPU is warmed up once and its timed run includes the copies. */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill(double *u, long n)
{
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);
}

static void jacobi_serial(double *u, long n, int iters)
{
    double *b = malloc(sizeof(double) * n * n);
    memcpy(b, u, sizeof(double) * (size_t)n * n);
    double *cur = u, *next = b;
    for (int it = 0; it < iters; it++) {
        for (long i = 1; i < n - 1; i++)
            for (long j = 1; j < n - 1; j++)
                next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] + cur[(i + 1) * n + j]
                                          + cur[i * n + j - 1] + cur[i * n + j + 1]);
        double *t = cur; cur = next; next = t;
    }
    if (cur != u)
        memcpy(u, cur, sizeof(double) * (size_t)n * n);
    free(b);
}

int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 120;
    double *u = malloc(sizeof(double) * n * n);
    double *r = malloc(sizeof(double) * n * n);   /* trusted serial result */
    if (!u || !r) { fprintf(stderr, "alloc failed\n"); return 1; }

    fill(r, n);
    double t0 = now();
    jacobi_serial(r, n, iters);
    double t_serial = now() - t0;

    fill(u, n);
    jacobi(u, n, 2);            /* GPU warm-up: context + JIT */
    fill(u, n);
    t0 = now();
    jacobi(u, n, iters);
    double t_gpu = now() - t0;

    /* the kernel must be RIGHT at the benchmark size, not just at the unit-test size:
     * compare every cell against the trusted serial result (the per-cell arithmetic is
     * identical, so correct kernels differ only by contraction noise ~1e-13; a kernel
     * that skips part of the grid differs by ~1) */
    double worst = 0.0;
    for (long t = 0; t < n * n; t++) {
        double d = u[t] - r[t];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    if (worst > 1e-6) {
        printf("BENCH CHECK FAIL: gpu result differs from the trusted serial result "
               "at n=%ld (worst cell difference %g)\n", n, worst);
        return 1;
    }

    printf("impl,time_s,speedup\n");
    printf("serial,%.6f,1.0\n", t_serial);
    printf("gpu,%.6f,%.3f\n", t_gpu, t_serial / t_gpu);
    free(u); free(r);
    return 0;
}
