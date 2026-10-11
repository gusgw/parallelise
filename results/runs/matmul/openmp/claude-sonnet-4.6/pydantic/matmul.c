/* matmul.c — Dense matrix multiply C = A * B for n x n row-major matrices.
 *
 * Parallelised with OpenMP: the outer loop over rows of C is distributed across
 * threads.  Each row is fully independent, so no reduction or critical section is
 * needed for the write to C.  The inner dot-product accumulator 's' is a private
 * scalar, so it is safe to accumulate without a reduction clause.
 */

#include <omp.h>
#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n)
{
    /* Parallelise over rows of C; each row i is independent. */
#pragma omp parallel for schedule(static)
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
