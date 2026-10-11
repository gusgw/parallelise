/*
 * matmul.cu  --  Dense matrix multiply C = A * B on the GPU.
 *
 * Strategy: 2-D grid of thread-blocks; each block owns a TILE x TILE sub-tile of C.
 * Threads cooperatively load tiles of A and B into shared memory and accumulate the
 * partial dot-products, giving O(n^3 / TILE) global-memory traffic instead of O(n^3).
 *
 * The host wrapper (extern "C" void matmul) has the EXACT signature from matmul.h:
 *   - allocates device buffers
 *   - copies A and B host -> device
 *   - launches the kernel
 *   - checks errors (cudaGetLastError + cudaDeviceSynchronize)
 *   - copies C device -> host
 *   - frees device buffers
 */

#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

/* Tile width.  16 gives 16*16*8 = 2 KB shared memory per matrix per block,
 * well within the 48 KB limit, and maps cleanly to warp width. */
#define TILE 16

/* --------------------------------------------------------------------------
 * Kernel: each thread computes one element C[row][col].
 * Grid is padded to a multiple of TILE; out-of-bounds threads do no work.
 * -------------------------------------------------------------------------- */
__global__ static void matmul_kernel(const double * __restrict__ A,
                                     const double * __restrict__ B,
                                     double       * __restrict__ C,
                                     long n)
{
    /* Shared-memory tiles for the current strip of A and B. */
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    /* Global row / column this thread is responsible for. */
    long row = (long)blockIdx.y * TILE + (long)threadIdx.y;
    long col = (long)blockIdx.x * TILE + (long)threadIdx.x;

    double acc = 0.0;

    /* Sweep over tiles along the shared k-dimension. */
    long nTiles = (n + TILE - 1) / TILE;
    for (long t = 0; t < nTiles; t++) {

        /* Load one element of A into shared memory (zero-pad if out of range). */
        long aCol = t * TILE + (long)threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? A[row * n + aCol] : 0.0;

        /* Load one element of B into shared memory (zero-pad if out of range). */
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

    /* Write result (guard against padded threads). */
    if (row < n && col < n)
        C[row * n + col] = acc;
}

/* --------------------------------------------------------------------------
 * Host function -- extern "C" so plain-C translation units can link against it.
 * -------------------------------------------------------------------------- */
extern "C"
void matmul(const double *A, const double *B, double *C, long n)
{
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    /* Allocate device memory. */
    double *dA = NULL, *dB = NULL, *dC = NULL;
    if (cudaMalloc((void **)&dA, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dB, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dC, bytes) != cudaSuccess) {
        fprintf(stderr, "matmul: cudaMalloc failed\n");
        exit(EXIT_FAILURE);
    }

    /* Copy inputs host -> device. */
    if (cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "matmul: cudaMemcpy H->D failed\n");
        exit(EXIT_FAILURE);
    }

    /* Launch: grid covers the full n x n output, padded to TILE multiples. */
    dim3 block(TILE, TILE, 1);
    dim3 grid((unsigned int)((n + TILE - 1) / TILE),
              (unsigned int)((n + TILE - 1) / TILE),
              1);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, n);

    /* Error checking. */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul: kernel launch error: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul: cudaDeviceSynchronize error: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy result device -> host. */
    if (cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "matmul: cudaMemcpy D->H failed\n");
        exit(EXIT_FAILURE);
    }

    /* Free device buffers. */
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
