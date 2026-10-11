#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scan.h"
#include "reference.h"
#include "problem.h"

int main(void)
{
    long n = REF_N;
    double *in = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));

    if (!in || !out) {
        fprintf(stderr, "Failed to allocate memory\n");
        free(in);
        free(out);
        return 1;
    }

    /* Fill input deterministically */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Run the scan */
    scan(in, out, n);

    /* Compute checksum */
    double cs = checksum(out, n);

    /* Check against reference */
    double err = fabs(cs - REF_CHECKSUM);
    int passed = (err <= REF_TOL);

    printf("checksum = %.16f\n", cs);
    printf("expected = %.16f\n", REF_CHECKSUM);
    printf("error    = %.16e\n", err);
    printf("tol      = %.16e\n", REF_TOL);
    printf("%s\n", passed ? "PASS" : "FAIL");

    free(in);
    free(out);

    return passed ? 0 : 1;
}
