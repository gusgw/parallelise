#include <stdlib.h>

#include "decay.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Reference unit test (validates the harness). Runs under any thread count; the check
 * checks it at 1 thread and at many -- the checksum must be the same. */
int main(void)
{
    long n = REF_N;
    double *w = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) w[i] = stim_of(i);

    decay(w, out, n);
    CHECK_CLOSE(checksum(out, n), REF_CHECKSUM, REF_TOL, "decay checksum matches the oracle");

    free(w); free(out);
    return TEST_SUMMARY();
}
