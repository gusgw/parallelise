#include <stdlib.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the check
 * checks it at 1 thread and at many -- the checksum must be the same. */
int main(void)
{
    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) in[i] = in_of(i);

    scan(in, out, n);
    CHECK_CLOSE(checksum(out, n), REF_CHECKSUM, REF_TOL, "scan checksum matches the oracle");

    free(in); free(out);
    return TEST_SUMMARY();
}
