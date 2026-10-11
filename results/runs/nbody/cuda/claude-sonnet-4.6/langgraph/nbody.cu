/*
 * nbody.cu  –  Direct 2-D N-body acceleration on the GPU.
 *
 * Strategy
 * --------
 * One thread owns one body i and accumulates its full acceleration by iterating
 * over every body j.  To amortise global-memory bandwidth the j-bodies are
 * loaded in shared-memory tiles of TILE_SIZE: every thread in the block
 * cooperates to fill the tile, then every thread walks the tile computing its
 * contribution.  This gives ~TILE_SIZE-fold reuse of each (px[j], py[j]) pair.
 *
 * Grid-stride loops handle n that is not a multiple of the grid size.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"

#define EPS2      0.0025          /* softening: 0.05^2                        */
#define TILE_SIZE 256             /* threads per block == tile width           */

/* -------------------------------------------------------------------------
 * Kernel: each thread computes ax[i], ay[i] for one body i.
 * Shared-memory tiles of j-bodies are used to reduce global traffic.
 * ------------------------------------------------------------------------- */
__global__ void nbody_kernel(const double * __restrict__ px,
                             const double * __restrict__ py,
                             double       * __restrict__ ax,
                             double       * __restrict__ ay,
                             long n)
{
    /* Shared tile: two arrays of TILE_SIZE doubles for px and py of j-bodies. */
    __shared__ double spx[TILE_SIZE];
    __shared__ double spy[TILE_SIZE];

    /* Grid-stride loop: each thread handles potentially multiple i-bodies. */
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (long)gridDim.x  * blockDim.x)
    {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0, sy = 0.0;

        /* Walk all j-bodies in tiles. */
        for (long tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
            /* Cooperatively load this tile into shared memory. */
            long j_load = tile_start + (long)threadIdx.x;
            if (j_load < n) {
                spx[threadIdx.x] = px[j_load];
                spy[threadIdx.x] = py[j_load];
            } else {
                /* Pad with a position that produces zero force on any body:
                 * set it to (xi, yi) so dx=dy=0, but we skip j==i anyway;
                 * actually any finite value works because we guard with j<n. */
                spx[threadIdx.x] = 0.0;
                spy[threadIdx.x] = 0.0;
            }
            __syncthreads();

            /* Accumulate contributions from this tile. */
            long tile_end = tile_start + TILE_SIZE;
            if (tile_end > n) tile_end = n;
            long tile_len = tile_end - tile_start;

            for (long t = 0; t < tile_len; t++) {
                long j = tile_start + t;
                if (j == i) continue;          /* skip self-interaction */
                double dx = spx[t] - xi;
                double dy = spy[t] - yi;
                double r2 = dx * dx + dy * dy;
                double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
                sx += dx / d3;
                sy += dy / d3;
            }
            __syncthreads();               /* protect tile before next load */
        }

        ax[i] = sx;
        ay[i] = sy;
    }
}

/* -------------------------------------------------------------------------
 * Host function (extern "C" so bench.c / test_nbody.c can link it as C).
 * Copies data to the device, launches the kernel, checks errors, copies back.
 * ------------------------------------------------------------------------- */
extern "C"
void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
    if (n <= 0) return;

    /* ---- device allocations ---- */
    double *d_px = NULL, *d_py = NULL, *d_ax = NULL, *d_ay = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    cudaMalloc((void **)&d_px, bytes);
    cudaMalloc((void **)&d_py, bytes);
    cudaMalloc((void **)&d_ax, bytes);
    cudaMalloc((void **)&d_ay, bytes);

    /* ---- copy inputs host → device ---- */
    cudaMemcpy(d_px, px, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, bytes, cudaMemcpyHostToDevice);

    /* ---- launch configuration ----
     * Blocks of TILE_SIZE threads; cap the grid so we don't over-subscribe
     * small GPUs, but allow enough blocks to keep the device busy.          */
    int threads = TILE_SIZE;
    long max_blocks = 1024;                          /* generous upper bound  */
    long blocks = (n + threads - 1) / threads;
    if (blocks > max_blocks) blocks = max_blocks;

    nbody_kernel<<<(int)blocks, threads>>>(d_px, d_py, d_ax, d_ay, n);

    /* ---- error checking ---- */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel launch error: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_ax); cudaFree(d_ay);
        return;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel sync error: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_ax); cudaFree(d_ay);
        return;
    }

    /* ---- copy results device → host ---- */
    cudaMemcpy(ax, d_ax, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, d_ay, bytes, cudaMemcpyDeviceToHost);

    /* ---- free device memory ---- */
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
