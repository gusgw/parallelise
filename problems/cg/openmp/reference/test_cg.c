#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the gate
 * checks it at 1 thread and at many -- the checksum must match the oracle either way. */
int main(void)
{
    long m = REF_M;
    double *x = malloc(sizeof(double) * m * m);
    double *b = malloc(sizeof(double) * m * m);
    for (long t = 0; t < m * m; t++) b[t] = b_of(t);

    cgsolve(x, b, m, REF_ITERS);
    CHECK_CLOSE(checksum(x, m), REF_CHECKSUM, REF_TOL, "cg checksum matches the oracle");

    free(x); free(b);
    return TEST_SUMMARY();
}
