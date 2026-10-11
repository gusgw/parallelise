#include "matmul.h"
#include <omp.h>

/* Dense matrix multiply C = A * B for n x n row-major matrices.
 * Parallelised with OpenMP over the rows of C. */
void matmul(const double *A, const double *B, double *C, long n)
{
    /* Each row of C is independent; parallelise over i */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            double s = 0.0;
            for (long k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = s;
        }
    }
}
