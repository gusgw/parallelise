// kernel_b.c – parallel implementation of the 2D stencil
// Implements the algorithm described in kernel_b.ipynb.
// The kernel updates interior points of an n×n grid for a given number of
// iterations using the 4‑point average. Boundaries remain unchanged.

#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kernel_b.h"

void kernel_b(double *u, long n, int iters)
{
    // Allocate a second buffer for the stencil updates.
    double *tmp = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!tmp) {
        // Allocation failure – abort.
        abort();
    }

    // We will ping‑pong between u (input) and tmp (output).
    // The pointer "in" always points to the current source data,
    // and "out" points to the destination for the next iteration.
    double *in = u;
    double *out = tmp;

    for (int it = 0; it < iters; ++it) {
        // Update interior points.
        #pragma omp parallel for schedule(static) collapse(2)
        for (long i = 1; i < n - 1; ++i) {
            for (long j = 1; j < n - 1; ++j) {
                out[i * n + j] = 0.25 * (in[(i - 1) * n + j] +
                                         in[(i + 1) * n + j] +
                                         in[i * n + (j - 1)] +
                                         in[i * n + (j + 1)]);
            }
        }

        // Copy boundary rows and columns (they stay unchanged).
        // Top and bottom rows.
        #pragma omp parallel for schedule(static)
        for (long j = 0; j < n; ++j) {
            out[j] = in[j];                                 // i = 0
            out[(n - 1) * n + j] = in[(n - 1) * n + j];     // i = n-1
        }
        // Left and right columns (including corners already handled).
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            out[i * n] = in[i * n];                         // j = 0
            out[i * n + (n - 1)] = in[i * n + (n - 1)];     // j = n-1
        }

        // Swap the role of the buffers for the next iteration.
        double *swap = in;
        in = out;
        out = swap;
    }

    // After the loop, "in" points to the buffer that holds the final data.
    // If the number of iterations is odd, the final data resides in tmp;
    // copy it back to the caller's array.
    if (iters % 2 == 1) {
        memcpy(u, in, (size_t)n * (size_t)n * sizeof(double));
    }
    // If iters is even, the data is already in u, nothing to do.

    free(tmp);
}
