/*
 * nbody.cu — Direct 2-D N-body acceleration on the GPU.
 *
 * Each CUDA thread owns one body i and accumulates the gravitational
 * acceleration from every other body j in a single pass (O(n^2) total work,
 * O(n) per thread).  A grid-stride loop lets the kernel handle any n with a
 * fixed launch configuration.
 *
 * The host wrapper (extern "C" nbody_accel) has the exact signature from
 * nbody.h: it receives host pointers, copies data to the device, launches the
 * kernel, synchronises, checks CUDA errors, copies results back, and frees
 * device memory.
 */

#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"

#define EPS2 0.0025   /* softening: 0.05 squared */

/* ------------------------------------------------------------------ kernel */

/*
 * Each thread computes ax[i] and ay[i] for one body i by iterating over all
 * bodies j.  A grid-stride outer loop handles n > gridDim.x * blockDim.x.
 *
 * We tile the j-loop through shared memory: each block loads a tile of
 * TILE_SIZE (px,py) pairs into shared memory so that the n reads of the
 * position arrays are coalesced and reused across threads in the same block.
 */
#define TILE_SIZE 256

__global__ void nbody_kernel(const double * __restrict__ px,
                             const double * __restrict__ py,
                             double       * __restrict__ ax,
                             double       * __restrict__ ay,
                             long n)
{
    /* shared-memory tile for the source bodies (j-side) */
    __shared__ double spx[TILE_SIZE];
    __shared__ double spy[TILE_SIZE];

    /* grid-stride loop: each thread processes one body i per stride */
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (long)gridDim.x * blockDim.x)
    {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0;
        double sy = 0.0;

        /* tile the j-loop through shared memory */
        for (long tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
            /* load tile: each thread in the block loads one element */
            long load_idx = tile_start + (long)threadIdx.x;
            if (load_idx < n) {
                spx[threadIdx.x] = px[load_idx];
                spy[threadIdx.x] = py[load_idx];
            }
            __syncthreads();

            /* accumulate contributions from this tile */
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
            __syncthreads();
        }

        ax[i] = sx;
        ay[i] = sy;
    }
}

/* --------------------------------------------------------------- host entry */

#ifdef __cplusplus
extern "C"
#endif
void nbody_accel(const double *px, const double *py,
                 double *ax, double *ay, long n)
{
    /* ---- allocate device buffers ---- */
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
     * blockDim = TILE_SIZE (matches the shared-memory tile width so that the
     * tile load is perfectly coalesced and every thread participates).
     * gridDim  = min(enough blocks to cover n, 1024) so we don't over-subscribe
     * the device for small n while still saturating it for large n.            */
    int block = TILE_SIZE;
    long grid_needed = (n + block - 1) / block;
    int  grid = (grid_needed < 1024L) ? (int)grid_needed : 1024;

    nbody_kernel<<<grid, block>>>(d_px, d_py, d_ax, d_ay, n);

    /* ---- error checking ---- */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel launch error: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_px); cudaFree(d_py);
        cudaFree(d_ax); cudaFree(d_ay);
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "nbody_kernel sync error: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_px); cudaFree(d_py);
        cudaFree(d_ax); cudaFree(d_ay);
        exit(EXIT_FAILURE);
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
