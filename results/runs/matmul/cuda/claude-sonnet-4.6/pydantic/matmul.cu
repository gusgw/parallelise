/*
 * matmul.cu  –  Dense matrix multiply C = A * B  (n x n, row-major, double)
 *
 * Strategy:
 *   • 2-D grid of 16×16 thread-blocks; each thread computes one C[i][j].
 *   • Shared-memory tiling (TILE×TILE) to reuse A-rows and B-columns from
 *     fast on-chip memory, reducing global-memory traffic by TILE×.
 *   • Grid-stride loops in both dimensions so the kernel handles any n.
 *   • Host wrapper (extern "C") owns all device allocation, H→D copies,
 *     kernel launch, error checking, D→H copy, and device-buffer free.
 */

#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

/* ------------------------------------------------------------------ */
/* Tile size – 16 gives 16×16 = 256 threads/block, a good occupancy   */
/* sweet-spot for FP64 on most NVIDIA architectures.                   */
/* ------------------------------------------------------------------ */
#define TILE 16

/* ------------------------------------------------------------------ */
/* Kernel                                                               */
/* ------------------------------------------------------------------ */
__global__ static void matmul_kernel(const double * __restrict__ A,
                                     const double * __restrict__ B,
                                     double       * __restrict__ C,
                                     long n)
{
    /* Shared-memory tiles for one strip of A and one strip of B */
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    /* This thread's output element */
    long row = (long)blockIdx.y * TILE + (long)threadIdx.y;
    long col = (long)blockIdx.x * TILE + (long)threadIdx.x;

    double acc = 0.0;

    /* Sweep over tiles along the k-dimension */
    long numTiles = (n + TILE - 1) / TILE;
    for (long t = 0; t < numTiles; t++) {

        /* Load one tile of A (row, k) */
        long kA = t * TILE + (long)threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < n && kA < n)
                                        ? A[row * n + kA] : 0.0;

        /* Load one tile of B (k, col) */
        long kB = t * TILE + (long)threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (kB < n && col < n)
                                        ? B[kB * n + col] : 0.0;

        __syncthreads();

        /* Accumulate dot-product for this tile */
        #pragma unroll
        for (int s = 0; s < TILE; s++)
            acc += sA[threadIdx.y][s] * sB[s][threadIdx.x];

        __syncthreads();
    }

    /* Write result */
    if (row < n && col < n)
        C[row * n + col] = acc;
}

/* ------------------------------------------------------------------ */
/* Host wrapper – exact signature from matmul.h                        */
/* ------------------------------------------------------------------ */
extern "C"
void matmul(const double *A, const double *B, double *C, long n)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    /* Allocate device buffers */
    double *dA = NULL, *dB = NULL, *dC = NULL;
    cudaError_t err;

    err = cudaMalloc((void **)&dA, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dA: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&dB, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dB: %s\n", cudaGetErrorString(err));
        cudaFree(dA);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&dC, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dC: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB);
        exit(EXIT_FAILURE);
    }

    /* Copy inputs host → device */
    err = cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy A H2D: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy B H2D: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Launch kernel: one thread per output element, tiled TILE×TILE blocks */
    dim3 block(TILE, TILE, 1);
    dim3 grid((unsigned int)((n + TILE - 1) / TILE),
              (unsigned int)((n + TILE - 1) / TILE),
              1);

    matmul_kernel<<<grid, block>>>(dA, dB, dC, n);

    /* Check for launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Wait for kernel to finish and check execution errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Copy result device → host */
    err = cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy C D2H: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Free device buffers */
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
