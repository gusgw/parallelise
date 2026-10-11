#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Checks the GPU solve's checksum against
 * the notebook oracle. */
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
