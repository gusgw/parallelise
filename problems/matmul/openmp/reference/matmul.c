#include "matmul.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Each row of C
 * is independent, so the outer loop parallelises directly. The k-then-j inner order
 * keeps the B accesses sequential in memory. */
void matmul(const double *A, const double *B, double *C, long n)
{
    #pragma omp parallel for
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
