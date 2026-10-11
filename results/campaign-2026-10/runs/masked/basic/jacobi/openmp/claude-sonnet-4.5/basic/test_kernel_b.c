#include <stdlib.h>
#include <math.h>
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    int iters = REF_ITERS;
    double *u = malloc(sizeof(double) * n * n);
    if (!u) return 1;

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    kernel_b(u, n, iters);

    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

    free(u);
    return TEST_SUMMARY();
}
