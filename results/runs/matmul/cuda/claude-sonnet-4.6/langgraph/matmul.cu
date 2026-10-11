/*
 * matmul.cu  –  Dense matrix multiply C = A * B on the GPU.
 *
 * Strategy: 2-D grid of thread-blocks; each block owns a (TILE x TILE) tile of C.
 * Threads cooperatively load tiles of A and B into shared memory, accumulate the
 * partial dot-products, then write the finished tile back to global memory.
 * This is the standard tiled GEMM pattern; it keeps global-memory traffic low and
 * achieves real GPU speedup over a single CPU core.
 */

#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

/* ------------------------------------------------------------------ */
/* Tile size.  16 gives 16*16 = 256 threads/block (a safe occupancy    */
/* sweet-spot for all CUDA compute capabilities >= 3.0).               */
/* ------------------------------------------------------------------ */
#define TILE 16

/* ------------------------------------------------------------------ */
/* Kernel                                                               */
/* ------------------------------------------------------------------ */
__global__ static void matmul_kernel(const double * __restrict__ A,
                                     const double * __restrict__ B,
                                           double * __restrict__ C,
                                     long n)
{
    /* Shared-memory tiles for the current strip of A and B. */
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    /* Global row / column this thread is responsible for. */
    long row = (long)blockIdx.y * TILE + (long)threadIdx.y;
    long col = (long)blockIdx.x * TILE + (long)threadIdx.x;

    double acc = 0.0;

    /* Sweep over all tiles along the shared k-dimension. */
    long nTiles = (n + TILE - 1) / TILE;
    for (long t = 0; t < nTiles; t++) {

        /* Load one element of A into shared memory (zero-pad if out of bounds). */
        long aCol = t * TILE + (long)threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? A[row * n + aCol] : 0.0;

        /* Load one element of B into shared memory (zero-pad if out of bounds). */
        long bRow = t * TILE + (long)threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0;

        __syncthreads();

        /* Accumulate the partial dot-product for this tile. */
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    /* Write result (guard against out-of-bounds threads for non-multiple-of-TILE n). */
    if (row < n && col < n)
        C[row * n + col] = acc;
}

/* ------------------------------------------------------------------ */
/* Host function – exact signature from matmul.h                       */
/* ------------------------------------------------------------------ */
extern "C"
void matmul(const double *A, const double *B, double *C, long n)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    /* Allocate device buffers. */
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

    /* Copy inputs host → device. */
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

    /* Launch kernel: one thread per output element, grouped into TILE×TILE blocks. */
    dim3 block(TILE, TILE, 1);
    dim3 grid((unsigned int)((n + TILE - 1) / TILE),
              (unsigned int)((n + TILE - 1) / TILE),
              1u);

    matmul_kernel<<<grid, block>>>(dA, dB, dC, n);

    /* Check for launch errors, then synchronise. */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Copy result device → host. */
    err = cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy C D2H: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        exit(EXIT_FAILURE);
    }

    /* Release device memory. */
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
