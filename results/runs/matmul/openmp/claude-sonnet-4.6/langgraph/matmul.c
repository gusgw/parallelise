/* matmul.c — Dense matrix multiply C = A * B for n x n row-major matrices.
 *
 * Parallelised with OpenMP: the outer loop over rows of C is distributed across
 * threads.  Each row is fully independent, so no reduction or synchronisation is
 * needed beyond the implicit barrier at the end of the parallel region.
 *
 * The inner k-loop is kept sequential inside each thread; the compiler can
 * auto-vectorise it freely.  A simple i-k-j loop order is used so that the
 * hot inner loop strides through B one element at a time (good cache behaviour
 * for the B column) while A[i][k] is a scalar broadcast.
 */

#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n)
{
    /* Parallelise over rows of C; each row is independent. */
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        /* Zero the output row first. */
        for (long j = 0; j < n; j++)
            C[i * n + j] = 0.0;

        /* Accumulate contributions from each column of A / row of B.
         * Loop order i-k-j: A[i*n+k] is a scalar; the inner j-loop
         * streams through a full row of B and a full row of C — both
         * contiguous in memory, so the hardware prefetcher and the
         * auto-vectoriser handle them well. */
        for (long k = 0; k < n; k++) {
            double a_ik = A[i * n + k];
            for (long j = 0; j < n; j++)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}
