/*
 * jacobi.cu  –  2-D Jacobi heat stencil, GPU implementation.
 *
 * Algorithm (from jacobi.ipynb):
 *   For `iters` sweeps, replace every INTERIOR cell by the average of its
 *   four neighbours using the PREVIOUS sweep's values (double-buffered).
 *   The four edges are fixed boundaries and are never modified.
 *
 *       u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *
 * The host function (extern "C" jacobi) copies the initial grid to the GPU,
 * alternates two device buffers for `iters` sweeps, copies the result back,
 * and frees the device buffers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

/* ------------------------------------------------------------------ */
/* Error-checking helper                                               */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ------------------------------------------------------------------ */
/* Kernel: one Jacobi sweep over the interior                          */
/*                                                                     */
/* Each thread handles one or more interior cells via a 2-D grid-     */
/* stride loop.  Boundaries (row 0, row n-1, col 0, col n-1) are      */
/* never written.                                                      */
/* ------------------------------------------------------------------ */
__global__ static void jacobi_sweep(const double * __restrict__ src,
                                          double * __restrict__ dst,
                                          long n)
{
    /* 2-D grid-stride loop over interior indices [1 .. n-2] x [1 .. n-2] */
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
/* Host entry point (extern "C" so bench.c / test_jacobi.c can link)  */
/* ------------------------------------------------------------------ */
extern "C"
void jacobi(double *u, long n, int iters)
{
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;

    /* Allocate two device buffers */
    double *d_cur = NULL, *d_nxt = NULL;
    CUDA_CHECK(cudaMalloc(&d_cur, bytes));
    CUDA_CHECK(cudaMalloc(&d_nxt, bytes));

    /* Copy the initial grid (including fixed boundary values) to both
     * buffers so that boundary rows/cols are already correct in d_nxt. */
    CUDA_CHECK(cudaMemcpy(d_cur, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nxt, u, bytes, cudaMemcpyHostToDevice));

    /* Choose a 2-D thread block; 16x16 = 256 threads is a safe default. */
    dim3 block(16, 16);
    /* Cover the interior; clamp grid to hardware limits (65535 per dim). */
    long interior = n - 2;
    if (interior <= 0) {
        /* Nothing to do – just free and return */
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaFree(d_nxt));
        return;
    }
    unsigned int gx = (unsigned int)((interior + block.x - 1) / block.x);
    unsigned int gy = (unsigned int)((interior + block.y - 1) / block.y);
    if (gx > 65535) gx = 65535;
    if (gy > 65535) gy = 65535;
    dim3 grid(gx, gy);

    /* Sweep loop: kernel reads d_cur, writes interior of d_nxt;
     * boundaries in d_nxt are already correct from the initial copy. */
    for (int it = 0; it < iters; it++) {
        jacobi_sweep<<<grid, block>>>(d_cur, d_nxt, n);

        /* Swap pointers – next iteration reads what we just wrote */
        double *tmp = d_cur;
        d_cur = d_nxt;
        d_nxt = tmp;
    }

    /* Check for kernel launch errors and wait for completion */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back (d_cur now holds the final grid after the last swap) */
    CUDA_CHECK(cudaMemcpy(u, d_cur, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_nxt));
}
