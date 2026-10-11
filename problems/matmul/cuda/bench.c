#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matmul.h"
#include "problem.h"

/* Trusted benchmark (agents do not touch this). CUDA has no worker-count sweep, so the
 * scaling evidence is serial CPU vs GPU on the same n (argv[1], default 1024):
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

static void matmul_serial(const double *A, const double *B, double *C, long n)
{
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++)
            C[i * n + j] = 0.0;
        for (long k = 0; k < n; k++) {
            const double a = A[i * n + k];
            for (long j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
    }
}

int main(int argc, char **argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 1024L;
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);
    double *R = malloc(sizeof(double) * n * n);   /* trusted serial result */
    if (!A || !B || !C || !R) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }

    double t0 = now();
    matmul_serial(A, B, R, n);
    double t_serial = now() - t0;

    matmul(A, B, C, n < 64 ? n : 64);   /* GPU warm-up: context + JIT */
    t0 = now();
    matmul(A, B, C, n);
    double t_gpu = now() - t0;

    /* the kernel must be RIGHT at the benchmark size, not just at the unit-test size:
     * compare every entry against the trusted serial result (entries are exact
     * integers in double, so correct kernels agree exactly; garbage does not) */
    double worst = 0.0;
    for (long t = 0; t < n * n; t++) {
        double d = C[t] - R[t];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    if (worst > 1e-6) {
        printf("BENCH CHECK FAIL: gpu result differs from the trusted serial result "
               "at n=%ld (worst entry difference %g)\n", n, worst);
        return 1;
    }

    printf("impl,time_s,speedup\n");
    printf("serial,%.6f,1.0\n", t_serial);
    printf("gpu,%.6f,%.3f\n", t_gpu, t_serial / t_gpu);
    free(A); free(B); free(C); free(R);
    return 0;
}
