#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cg.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). CUDA has no worker-count sweep, so the
 * scaling evidence is serial CPU vs GPU on the same problem (m = argv[1] default 1024,
 * iters = argv[2] default 100):
 *     impl,time_s,speedup   with rows  serial,<t>,1.0  and  gpu,<t>,<t_serial/t_gpu>
 * The serial baseline lives HERE, in trusted code; cgsolve starts from x = 0 so each
 * measurement solves the identical system; the GPU is warmed up once and its timed run
 * includes the host-device copies -- that is what a caller actually gets. */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double dot_serial(const double *u, const double *v, long n)
{
    double s = 0.0;
    for (long k = 0; k < n; k++) s += u[k] * v[k];
    return s;
}

static void cgsolve_serial(double *x, const double *b, long m, int iters)
{
    const long n = m * m;
    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);
    for (long k = 0; k < n; k++) { x[k] = 0.0; r[k] = b[k]; p[k] = b[k]; }
    double rr = dot_serial(r, r, n);
    for (int it = 0; it < iters; it++) {
        for (long i = 0; i < m; i++)
            for (long j = 0; j < m; j++) {
                const long idx = i * m + j;
                double s = 4.0 * p[idx];
                if (i > 0)     s -= p[idx - m];
                if (i < m - 1) s -= p[idx + m];
                if (j > 0)     s -= p[idx - 1];
                if (j < m - 1) s -= p[idx + 1];
                q[idx] = s;
            }
        const double alpha = rr / dot_serial(p, q, n);
        for (long k = 0; k < n; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }
        const double rr_new = dot_serial(r, r, n);
        const double beta = rr_new / rr;
        rr = rr_new;
        for (long k = 0; k < n; k++)
            p[k] = r[k] + beta * p[k];
    }
    free(r); free(p); free(q);
}

int main(int argc, char **argv)
{
    long m = (argc > 1) ? atol(argv[1]) : 1024L;
    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    double *R = malloc(sizeof(double) * m * m);   /* trusted serial result */
    if (!x || !b || !R) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long t = 0; t < m * m; t++) b[t] = b_of(t);

    double t0 = now();
    cgsolve_serial(R, b, m, iters);
    double t_serial = now() - t0;

    cgsolve(x, b, m < 64 ? m : 64, 2);   /* GPU warm-up: context + JIT */
    t0 = now();
    cgsolve(x, b, m, iters);
    double t_gpu = now() - t0;

    /* the kernel must be RIGHT at the benchmark size, not just at the unit-test size:
     * compare every entry against the trusted serial result. The GPU's dot products
     * legitimately sum in a different order and CG carries that difference through all
     * the iterations, so we allow a small RELATIVE disagreement (scaled by the largest
     * entry); a kernel that skips part of the grid or never converges is off by order
     * one relative and fails. */
    double worst = 0.0, biggest = 0.0;
    for (long t = 0; t < m * m; t++) {
        double d = x[t] - R[t];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        double a = R[t] < 0 ? -R[t] : R[t];
        if (a > biggest) biggest = a;
    }
    if (worst > 1e-6 * biggest) {
        printf("BENCH CHECK FAIL: gpu result differs from the trusted serial result "
               "at m=%ld (worst entry difference %g, largest entry %g)\n",
               m, worst, biggest);
        return 1;
    }

    printf("impl,time_s,speedup\n");
    printf("serial,%.6f,1.0\n", t_serial);
    printf("gpu,%.6f,%.3f\n", t_gpu, t_serial / t_gpu);
    free(x); free(b); free(R);
    return 0;
}
