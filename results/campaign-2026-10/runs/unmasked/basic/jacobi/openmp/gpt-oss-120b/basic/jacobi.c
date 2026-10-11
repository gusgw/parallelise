#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters) {
    if (n <= 2 || iters <= 0) return; // nothing to do
    double *new_u = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!new_u) return; // allocation failure, silently return
    // Preserve the original pointer for final copy if needed
    double *orig = u;
    double *cur = u;
    double *next = new_u;

    for (int it = 0; it < iters; ++it) {
        // copy fixed boundaries from cur to next (top row, bottom row, left/right columns)
        // top row (i == 0)
        memcpy(&next[0], &cur[0], (size_t)n * sizeof(double));
        // bottom row (i == n-1)
        memcpy(&next[(n - 1) * n], &cur[(n - 1) * n], (size_t)n * sizeof(double));
        // left and right columns for interior rows
        #pragma omp parallel for
        for (long i = 1; i < n - 1; ++i) {
            next[i * n + 0] = cur[i * n + 0];
            next[i * n + (n - 1)] = cur[i * n + (n - 1)];
        }

        // interior update
        #pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; ++i) {
            long base = i * n;
            for (long j = 1; j < n - 1; ++j) {
                next[base + j] = 0.25 * (cur[base - n + j]   // i-1, j
                                         + cur[base + n + j]   // i+1, j
                                         + cur[base + j - 1]   // i, j-1
                                         + cur[base + j + 1]); // i, j+1
            }
        }
        // swap cur and next
        double *tmp = cur;
        cur = next;
        next = tmp;
    }

    // Ensure final result resides in the original array
    if (cur != orig) {
        memcpy(orig, cur, (size_t)n * (size_t)n * sizeof(double));
    }
    free(new_u);
}
