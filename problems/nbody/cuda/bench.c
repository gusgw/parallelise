#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nbody.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). CUDA has no worker-count sweep, so the
 * scaling evidence is serial CPU vs GPU on the same n (argv[1], default 32768):
 *     impl,time_s,speedup   with rows  serial,<t>,1.0  and  gpu,<t>,<t_serial/t_gpu>
 * The serial baseline lives HERE, in trusted code. The GPU is warmed up once before
 * timing (first call pays context costs); the timed GPU run includes the host-device
 * copies -- that is what a caller actually gets. */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void nbody_accel_serial(const double *px, const double *py,
                               double *ax, double *ay, long n)
{
    const double eps2 = 0.0025;   /* softening 0.05 squared */
    for (long i = 0; i < n; i++) {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0, sy = 0.0;
        for (long j = 0; j < n; j++) {
            if (j == i)
                continue;
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double r2 = dx * dx + dy * dy;
            const double d3 = (r2 + eps2) * sqrt(r2 + eps2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}

int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 32768L;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    double *rx = malloc(sizeof(double) * n);   /* trusted serial result */
    double *ry = malloc(sizeof(double) * n);
    if (!px || !py || !ax || !ay || !rx || !ry) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }

    double t0 = now();
    nbody_accel_serial(px, py, rx, ry, n);
    double t_serial = now() - t0;

    nbody_accel(px, py, ax, ay, n < 1024 ? n : 1024);   /* GPU warm-up: context + JIT */
    t0 = now();
    nbody_accel(px, py, ax, ay, n);
    double t_gpu = now() - t0;

    /* the kernel must be RIGHT at the benchmark size, not just at the unit-test size:
     * compare every acceleration against the trusted serial result (each is a plain sum
     * of ~n bounded terms in the same order, so correct kernels agree to ~1e-9 even
     * with FMA contraction on the device; garbage does not) */
    double worst = 0.0;
    for (long i = 0; i < n; i++) {
        double d = fabs(ax[i] - rx[i]);
        if (d > worst) worst = d;
        d = fabs(ay[i] - ry[i]);
        if (d > worst) worst = d;
    }
    if (worst > 1e-6) {
        printf("BENCH CHECK FAIL: gpu result differs from the trusted serial result "
               "at n=%ld (worst acceleration difference %g)\n", n, worst);
        return 1;
    }

    printf("impl,time_s,speedup\n");
    printf("serial,%.6f,1.0\n", t_serial);
    printf("gpu,%.6f,%.3f\n", t_gpu, t_serial / t_gpu);
    free(px); free(py); free(ax); free(ay); free(rx); free(ry);
    return 0;
}
