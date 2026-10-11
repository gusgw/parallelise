#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kernel_b.h"

/*
 * Parallel implementation of the 2D Jacobi stencil.
 * The input array `u` is a flattened n-by-n matrix stored in row-major order.
 * The kernel updates interior points for `iters` iterations, leaving the
 * boundary unchanged (the first row is initialized to 1.0, others to 0.0).
 *
 * The algorithm follows the reference Python implementation:
 *   for each iteration:
 *       copy current state to a temporary buffer
 *       update interior points in the temporary buffer using the current state
 *       swap buffers
 *   after the loop, ensure the final result resides in `u`.
 */

void kernel_b(double *u, long n, int iters)
{
    if (n <= 0 || iters <= 0) {
        return; /* nothing to do */
    }

    const size_t size = (size_t)n * (size_t)n;
    double *next = (double *)malloc(size * sizeof(double));
    if (!next) {
        /* allocation failure – abort the kernel */
        return;
    }

    double *cur = u;      /* cur points to the current state */
    double *tmp;

    for (int it = 0; it < iters; ++it) {
        /* Copy current state to the buffer that will receive updates */
        memcpy(next, cur, size * sizeof(double));

        /* Update interior points */
        #pragma omp parallel for collapse(2) schedule(static)
        for (long i = 1; i < n - 1; ++i) {
            for (long j = 1; j < n - 1; ++j) {
                size_t idx = (size_t)i * n + j;
                next[idx] = 0.25 * (cur[idx - n]      /* up */
                                 + cur[idx + n]      /* down */
                                 + cur[idx - 1]      /* left */
                                 + cur[idx + 1]);    /* right */
            }
        }

        /* Swap buffers for the next iteration */
        tmp = cur;
        cur = next;
        next = tmp;
    }

    /* If the final data resides in the auxiliary buffer, copy it back to `u`. */
    if (cur != u) {
        memcpy(u, cur, size * sizeof(double));
    }

    free(next);
}
