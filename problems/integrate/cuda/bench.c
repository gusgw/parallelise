#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "integrate.h"

/* Trusted benchmark (agents do not touch this). CUDA has no worker-count sweep, so the
 * scaling evidence is serial CPU vs GPU on the same n: prints
 *     impl,time_s,speedup
 *     serial,<t>,1.0
 *     gpu,<t>,<t_serial/t_gpu>
 * n is argv[1] (default 2e8). The serial baseline lives HERE, in trusted code, so the
 * agent's kernel cannot influence it. The GPU is warmed up once before timing (first
 * call pays context/JIT costs). Wall time via clock_gettime(CLOCK_MONOTONIC). */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double integrate_serial(double a, double b, long n)
{
    const double h = (b - a) / (double)n;
    double s = 0.0;
    for (long i = 0; i < n; i++) {
        const double x = a + ((double)i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }
    return s * h;
}

int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 200000000L;

    double t0 = now();
    double rs = integrate_serial(0.0, 1.0, n);
    double t_serial = now() - t0;

    volatile double warm = integrate(0.0, 1.0, 1000L); /* GPU warm-up: context + JIT */
    (void)warm;
    t0 = now();
    double rg = integrate(0.0, 1.0, n);
    double t_gpu = now() - t0;

    /* the kernel must be RIGHT at the benchmark size, not just at the unit-test size:
     * correct implementations differ from the trusted serial value only by summation
     * order (~1e-12); a kernel that skips part of the range is off by far more */
    double diff = rg - rs;
    if (diff < 0) diff = -diff;
    if (diff > 1e-6) {
        printf("BENCH CHECK FAIL: gpu result %.15f differs from the trusted serial "
               "result %.15f at n=%ld (diff %g)\n", rg, rs, n, diff);
        return 1;
    }

    printf("impl,time_s,speedup\n");
    printf("serial,%.6f,1.0\n", t_serial);
    printf("gpu,%.6f,%.3f\n", t_gpu, t_serial / t_gpu);
    return 0;
}
