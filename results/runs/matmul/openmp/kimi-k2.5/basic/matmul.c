#include "matmul.h"
#include <omp.h>

/* Dense matrix multiply C = A * B for n x n row-major matrices.
 * Parallelised with OpenMP over the rows of C. */
void matmul(const double *A, const double *B, double *C, long n)
{
    /* Parallelise over rows (i). Each row of C is independent.
     * Use collapse(2) to parallelise both i and j loops for better load balancing
     * when n is small, but keep it simple with just i-loop parallelisation
     * which is sufficient for typical matrix sizes. */
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
