/*
 * nbody.cu — Direct 2-D N-body acceleration on the GPU.
 *
 * Each thread owns one body i and accumulates the gravitational acceleration
 * from every other body j.  The j-loop is tiled: each block cooperatively
 * loads a tile of TILE bodies into shared memory so the (px[j], py[j]) values
 * are read from L1/shared rather than global memory for every i in the block.
 *
 * Implements the signature declared in nbody.h (extern "C").
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"

#define EPS2   0.0025          /* softening: 0.05 squared                    */
#define TILE   256             /* shared-memory tile width (bodies per tile)  */

/* -------------------------------------------------------------------------
 * Kernel: one thread per body i, tiled j-loop via shared memory.
 * -------------------------------------------------------------------------*/
__global__ static void nbody_kernel(const double * __restrict__ dpx,
                                    const double * __restrict__ dpy,
                                    double       * __restrict__ dax,
                                    double       * __restrict__ day,
                                    long n)
{
    /* Shared-memory tile: interleaved x/y to keep the two loads together.   */
    __shared__ double spx[TILE];
    __shared__ double spy[TILE];

    /* Grid-stride loop: each thread handles one or more body indices i.      */
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (long)gridDim.x  * blockDim.x)
    {
        const double xi = dpx[i];
        const double yi = dpy[i];
        double sx = 0.0, sy = 0.0;

        /* Tile the j-loop so every warp reads shared memory, not global.    */
        for (long tile_start = 0; tile_start < n; tile_start += TILE) {
            /* Cooperatively load this tile.                                  */
            long load_idx = tile_start + (long)threadIdx.x;
            if (load_idx < n) {
                spx[threadIdx.x] = dpx[load_idx];
                spy[threadIdx.x] = dpy[load_idx];
            } else {
                /* Pad with a position that produces zero force on any body:
                 * place the ghost at (xi, yi) so dx=dy=0 and d3 is just
                 * EPS2^(3/2) — the contribution is exactly 0/d3 = 0.        */
                spx[threadIdx.x] = xi;
                spy[threadIdx.x] = yi;
            }
            __syncthreads();

            /* Accumulate contributions from this tile.                       */
            long tile_end = tile_start + TILE;
            if (tile_end > n) tile_end = n;
            long tile_len = tile_end - tile_start;

            for (long t = 0; t < tile_len; t++) {
                long j = tile_start + t;
                double dx = spx[t] - xi;
                double dy = spy[t] - yi;
                double r2 = dx * dx + dy * dy;
                double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
                /* Skip self (j == i): dx=dy=0, contribution is 0/d3 = 0.
                 * We still execute the division to avoid warp divergence;
                 * the result is 0.0 / d3 = 0.0 which is correct.            */
                double contrib = (j == i) ? 0.0 : 1.0;
                sx += contrib * dx / d3;
                sy += contrib * dy / d3;
            }
            __syncthreads();
        }

        dax[i] = sx;
        day[i] = sy;
    }
}

/* -------------------------------------------------------------------------
 * Host function (extern "C" so plain-C callers can link against it).
 * -------------------------------------------------------------------------*/
extern "C"
void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
    if (n <= 0) return;

    const size_t bytes = (size_t)n * sizeof(double);

    /* Allocate device buffers. */
    double *dpx = NULL, *dpy = NULL, *dax = NULL, *day = NULL;
    cudaMalloc((void **)&dpx, bytes);
    cudaMalloc((void **)&dpy, bytes);
    cudaMalloc((void **)&dax, bytes);
    cudaMalloc((void **)&day, bytes);

    /* Copy inputs host → device. */
    cudaMemcpy(dpx, px, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dpy, py, bytes, cudaMemcpyHostToDevice);

    /* Launch: one thread per body, block size = TILE so the shared-memory
     * tile is exactly one block's worth of threads.
     * Cap the grid to 65535 blocks (hardware limit) — the grid-stride loop
     * handles any remainder automatically.                                   */
    int block = TILE;
    long grid_long = (n + block - 1) / block;
    int  grid = (grid_long > 65535L) ? 65535 : (int)grid_long;

    nbody_kernel<<<grid, block>>>(dpx, dpy, dax, day, n);

    /* Check for launch errors and wait for completion. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel launch error: %s\n",
                cudaGetErrorString(err));
        cudaFree(dpx); cudaFree(dpy); cudaFree(dax); cudaFree(day);
        return;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel sync error: %s\n",
                cudaGetErrorString(err));
        cudaFree(dpx); cudaFree(dpy); cudaFree(dax); cudaFree(day);
        return;
    }

    /* Copy results device → host. */
    cudaMemcpy(ax, dax, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, day, bytes, cudaMemcpyDeviceToHost);

    /* Free device buffers. */
    cudaFree(dpx);
    cudaFree(dpy);
    cudaFree(dax);
    cudaFree(day);
}
