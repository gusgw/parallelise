#include <stdlib.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the gate
 * checks it at 1 thread and at many -- the checksum must be the same. */
int main(void)
{
    long n = REF_N;
    double *u = malloc(sizeof(double) * n * n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) u[i * n + j] = u_init(i, j, n);

    jacobi(u, n, REF_ITERS);
    CHECK_CLOSE(checksum(u, n), REF_CHECKSUM, REF_TOL, "jacobi checksum matches the oracle");

    free(u);
    return TEST_SUMMARY();
}
