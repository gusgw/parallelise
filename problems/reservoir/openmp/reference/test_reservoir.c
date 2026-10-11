#include <stdlib.h>

#include "reservoir.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the gate
 * checks it at 1 thread and at many -- the checksum must be the same. */
int main(void)
{
    long n = REF_N;
    double *a = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) a[i] = change_of(i);

    reservoir(a, out, n);
    CHECK_CLOSE(checksum(out, n), REF_CHECKSUM, REF_TOL,
                "reservoir checksum matches the oracle");

    free(a); free(out);
    return TEST_SUMMARY();
}
