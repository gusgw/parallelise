#include <stdlib.h>
#include <string.h>
#include "kernel_b.h"

void kernel_b(double *u, long n, int iters)
{
    // Allocate a second buffer for the Jacobi updates.
    double *buf = (double *)malloc(n * n * sizeof(double));
    if (!buf) {
        abort();
    }

    double *cur = u;   // current state
    double *next = buf; // next state

    for (int it = 0; it < iters; ++it) {
        // Copy boundary rows (i = 0 and i = n-1).
        #pragma omp parallel for
        for (long j = 0; j < n; ++j) {
            next[j] = cur[j]; // row 0
            next[(n - 1) * n + j] = cur[(n - 1) * n + j]; // row n-1
        }
        // Copy boundary columns for interior rows.
        #pragma omp parallel for
        for (long i = 1; i < n - 1; ++i) {
            next[i * n] = cur[i * n];                 // column 0
            next[i * n + (n - 1)] = cur[i * n + (n - 1)]; // column n-1
        }

        // Update interior points.
        #pragma omp parallel for collapse(2)
        for (long i = 1; i < n - 1; ++i) {
            for (long j = 1; j < n - 1; ++j) {
                next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] +
                                          cur[(i + 1) * n + j] +
                                          cur[i * n + (j - 1)] +
                                          cur[i * n + (j + 1)]);
            }
        }
        // Swap buffers.
        double *tmp = cur;
        cur = next;
        next = tmp;
    }

    // Ensure the final result resides in the original array 'u'.
    if (cur != u) {
        memcpy(u, cur, n * n * sizeof(double));
    }
    free(buf);
}
