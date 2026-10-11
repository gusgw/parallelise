/*
 * jacobi.cu  –  2-D Jacobi heat stencil, GPU implementation.
 *
 * Algorithm (from jacobi.ipynb):
 *   Grid: n x n, row-major.  Top row (i==0) is hot (1.0); all other edges cold (0.0).
 *   Each sweep: u_new[i][j] = 0.25*(u[i-1][j]+u[i+1][j]+u[i][j-1]+u[i][j+1])
 *   for every INTERIOR cell (1 <= i,j <= n-2).  Boundaries are never modified.
 *   Double-buffered: kernel reads one device buffer, writes the other; they are
 *   ping-ponged each sweep.  Boundaries are copied once into both buffers before
 *   the loop starts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

/* ------------------------------------------------------------------ helpers */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ------------------------------------------------------------------ kernel  */

/*
 * Each thread is responsible for exactly one interior cell (ci, cj) where
 *   ci = blockIdx.y * blockDim.y + threadIdx.y + 1   (1 … n-2)
 *   cj = blockIdx.x * blockDim.x + threadIdx.x + 1   (1 … n-2)
 *
 * Grid-stride loops let the same kernel handle any n without re-launching with
 * different configs.
 */
__global__ static void jacobi_sweep(const double * __restrict__ src,
                                          double * __restrict__ dst,
                                          long n)
{
    /* grid-stride over interior rows */
    for (long i = (long)blockIdx.y * blockDim.y + threadIdx.y + 1;
         i < n - 1;
         i += (long)gridDim.y * blockDim.y)
    {
        /* grid-stride over interior columns */
        for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x + 1;
             j < n - 1;
             j += (long)gridDim.x * blockDim.x)
        {
            dst[i * n + j] = 0.25 * (src[(i - 1) * n + j] +
                                     src[(i + 1) * n + j] +
                                     src[ i * n + j - 1 ] +
                                     src[ i * n + j + 1 ]);
        }
    }
}

/* ------------------------------------------------------------------ host API */

extern "C"
void jacobi(double *u, long n, int iters)
{
    if (n < 2 || iters <= 0) return;

    const size_t bytes = sizeof(double) * (size_t)n * (size_t)n;

    /* Allocate two device buffers */
    double *d_a = NULL, *d_b = NULL;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    /* Copy the initial grid (with boundaries) into both buffers so that
     * whichever buffer is "dst" on the first sweep already has correct
     * boundary values – they are never touched by the kernel again.        */
    CUDA_CHECK(cudaMemcpy(d_a, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, u, bytes, cudaMemcpyHostToDevice));

    /* Launch configuration: 16×16 thread blocks cover the interior.
     * Grid dimensions are capped at the device maximum (65535) but for
     * typical n the interior fits in one grid pass; the kernel uses
     * grid-stride loops for safety.                                        */
    const int TILE = 16;
    long interior = n - 2;                          /* number of interior rows/cols */
    if (interior <= 0) {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        return;
    }

    dim3 block(TILE, TILE);
    dim3 grid((unsigned int)((interior + TILE - 1) / TILE),
              (unsigned int)((interior + TILE - 1) / TILE));

    /* Cap grid dims to hardware limit (grid-stride handles the rest) */
    {
        int maxGrid = 65535;
        if ((int)grid.x > maxGrid) grid.x = (unsigned int)maxGrid;
        if ((int)grid.y > maxGrid) grid.y = (unsigned int)maxGrid;
    }

    double *src = d_a, *dst = d_b;

    for (int it = 0; it < iters; it++) {
        jacobi_sweep<<<grid, block>>>(src, dst, n);
        /* swap pointers */
        double *tmp = src; src = dst; dst = tmp;
    }

    /* Check for kernel errors */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back – after the last swap, 'src' holds the final grid */
    CUDA_CHECK(cudaMemcpy(u, src, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}
