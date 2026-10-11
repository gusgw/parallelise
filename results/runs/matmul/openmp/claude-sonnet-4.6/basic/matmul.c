/* matmul.c -- Dense matrix multiply C = A * B for n x n row-major matrices.
 * Parallelised with OpenMP: the outer loop over rows of C is distributed across
 * threads; each row is fully independent so no reduction or synchronisation is
 * needed beyond the implicit barrier at the end of the parallel region.
 * The inner k-loop accumulates into a thread-local scalar 's', avoiding any
 * shared-write hazard. */

#include <omp.h>
#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n)
{
    /* Parallelise over rows of C.  Each iteration writes exactly one row of C
     * (C[i*n .. i*n+n-1]) and reads one row of A and all of B -- no overlap
     * between iterations, so 'schedule(static)' is safe and efficient. */
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
