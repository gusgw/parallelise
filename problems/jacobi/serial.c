/* The serial routine of jacobi.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 8, 64, 256, 1024]. */
#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters) {
    if (n < 3 || iters <= 0) {
        return;
    }
    
    // Allocate a temporary array for the new values
    double *u_new = malloc(n * n * sizeof(double));
    if (u_new == NULL) {
        return;
    }
    
    // Perform Jacobi iterations
    for (int iter = 0; iter < iters; iter++) {
        // Copy current u to u_new (to preserve boundary values)
        memcpy(u_new, u, n * n * sizeof(double));
        
        // Update interior points in u_new based on values from u
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                // u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
                u_new[i * n + j] = 0.25 * (
                    u[(i - 1) * n + j] +
                    u[(i + 1) * n + j] +
                    u[i * n + (j - 1)] +
                    u[i * n + (j + 1)]
                );
            }
        }
        
        // Copy u_new back to u for next iteration
        memcpy(u, u_new, n * n * sizeof(double));
    }
    
    free(u_new);
}
