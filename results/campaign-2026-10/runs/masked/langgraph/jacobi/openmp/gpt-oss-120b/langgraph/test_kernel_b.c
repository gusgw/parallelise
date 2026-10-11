#include <stdlib.h>
#include "testing.h"
#include "kernel_b.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    size_t size = (size_t)n * (size_t)n;
    double *u = (double *)malloc(size * sizeof(double));
    if (!u) {
        abort();
    }

    // Initialize the grid using the problem's initializer.
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    // Run the kernel.
    kernel_b(u, n, iters);

    // Compute checksum and verify against reference.
    double chk = checksum(u, n);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum mismatch");

    free(u);
    return TEST_SUMMARY();
}
