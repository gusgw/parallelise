/*
 * jacobi.cu  --  2-D Jacobi heat stencil, GPU implementation.
 *
 * Algorithm (from jacobi.ipynb):
 *   Solve steady-state heat on an n x n plate by Jacobi iteration.
 *   Top edge is hot (1.0), the other three edges cold (0.0).
 *   Each sweep replaces every interior cell by the average of its four
 *   neighbours using the PREVIOUS sweep's values (double-buffered):
 *       u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *   The four edges are FIXED boundaries and are never modified.
 *   After `iters` sweeps the final grid is in u.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/* Kernel: one Jacobi sweep over the interior cells                     */
/* ------------------------------------------------------------------ */

/*
 * Each thread handles one or more interior cells via a 2-D grid-stride loop.
 * The kernel reads from `src` and writes the interior of `dst`.
 * Boundary rows/columns in `dst` are already initialised (copied once on the
 * host before the iteration loop) and are never touched here.
 */
__global__ void jacobi_sweep(const double * __restrict__ src,
                              double       * __restrict__ dst,
                              long n)
{
    /* grid-stride loop over interior indices [1 .. n-2] x [1 .. n-2] */
    for (long i = (long)blockIdx.y * blockDim.y + threadIdx.y + 1;
         i < n - 1;
         i += (long)gridDim.y * blockDim.y)
    {
        for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x + 1;
             j < n - 1;
             j += (long)gridDim.x * blockDim.x)
        {
            dst[i * n + j] = 0.25 * (src[(i - 1) * n + j] +
                                     src[(i + 1) * n + j] +
                                     src[i * n + (j - 1)] +
                                     src[i * n + (j + 1)]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Host entry point (extern "C" so test_jacobi.c can link it)          */
/* ------------------------------------------------------------------ */

extern "C"
void jacobi(double *u, long n, int iters)
{
    if (iters <= 0 || n <= 0) return;

    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;

    /* Allocate two device buffers */
    double *d_cur = NULL, *d_next = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_cur,  bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_next, bytes));

    /* Copy the initial grid (including fixed boundaries) to both buffers.
     * Copying to both means d_next already has the correct boundary values
     * before the first sweep; we never write boundaries in the kernel. */
    CUDA_CHECK(cudaMemcpy(d_cur,  u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, u, bytes, cudaMemcpyHostToDevice));

    /* Launch configuration: 16x16 thread blocks, grid covers interior */
    dim3 block(16, 16);
    long interior = (n > 2) ? (n - 2) : 0;
    dim3 grid((unsigned int)((interior + block.x - 1) / block.x),
              (unsigned int)((interior + block.y - 1) / block.y));

    /* Cap grid dimensions to device limits (handles very large n) */
    {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (grid.x > (unsigned int)prop.maxGridSize[0])
            grid.x = (unsigned int)prop.maxGridSize[0];
        if (grid.y > (unsigned int)prop.maxGridSize[1])
            grid.y = (unsigned int)prop.maxGridSize[1];
    }

    /* Sweep loop: alternate src/dst each iteration */
    for (int it = 0; it < iters; it++) {
        jacobi_sweep<<<grid, block>>>(d_cur, d_next, n);

        /* swap pointers */
        double *tmp = d_cur;
        d_cur  = d_next;
        d_next = tmp;
    }

    /* Check for kernel errors and synchronise */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host (d_cur holds the final grid after last swap) */
    CUDA_CHECK(cudaMemcpy(u, d_cur, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
}
