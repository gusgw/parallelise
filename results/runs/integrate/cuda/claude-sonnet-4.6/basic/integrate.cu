/*
 * integrate.cu — GPU midpoint-rule estimate of ∫ 4/(1+x²) dx over [a, b].
 *
 * Algorithm (from integrate.ipynb):
 *   h = (b - a) / n
 *   sum_{i=0}^{n-1}  4 / (1 + (a + (i + 0.5)*h)^2)  *  h
 *
 * Implementation:
 *   • A grid-stride __global__ kernel: each thread accumulates its share of
 *     the n subintervals into a register partial sum.
 *   • A shared-memory per-block reduction collapses the block's partials into
 *     one double that is written to a device array d_block[blockIdx.x].
 *   • The host function sums the per-block results and returns the integral.
 */

#include <stdio.h>
#include <stdlib.h>

#include "integrate.h"

/* ------------------------------------------------------------------ kernel */

/*
 * Each thread strides over [0, n) with step gridDim.x * blockDim.x,
 * accumulating its partial sum.  A shared-memory tree reduction then
 * produces one double per block, stored in d_block[blockIdx.x].
 */
__global__ static void integrate_kernel(double a, double h, long n,
                                        double *d_block)
{
    /* Dynamic shared memory: blockDim.x doubles */
    extern __shared__ double sdata[];

    long tid   = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    long stride = (long)gridDim.x  * (long)blockDim.x;

    /* Each thread accumulates its own partial sum (register) */
    double partial = 0.0;
    for (long i = tid; i < n; i += stride) {
        double x = a + ((double)i + 0.5) * h;
        partial += 4.0 / (1.0 + x * x);
    }

    /* Load into shared memory */
    sdata[threadIdx.x] = partial;
    __syncthreads();

    /* Tree reduction within the block */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes the block result */
    if (threadIdx.x == 0) {
        d_block[blockIdx.x] = sdata[0];
    }
}

/* --------------------------------------------------------------- host entry */

#ifdef __cplusplus
extern "C"
#endif
double integrate(double a, double b, long n)
{
    if (n <= 0) return 0.0;

    const double h = (b - a) / (double)n;

    /* Choose launch geometry */
    const int BLOCK = 256;
    /* Cap grid so we don't allocate more blocks than needed */
    long max_blocks = (n + (long)BLOCK - 1L) / (long)BLOCK;
    int  GRID = (max_blocks < 1024L) ? (int)max_blocks : 1024;

    /* Allocate per-block partial-sum array on device */
    double *d_block = NULL;
    cudaError_t err = cudaMalloc((void **)&d_block, (size_t)GRID * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "integrate: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return 0.0;
    }

    /* Launch kernel */
    size_t smem = (size_t)BLOCK * sizeof(double);
    integrate_kernel<<<GRID, BLOCK, smem>>>(a, h, n, d_block);

    /* Check for launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "integrate: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_block);
        return 0.0;
    }

    /* Wait for kernel to finish */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "integrate: cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_block);
        return 0.0;
    }

    /* Copy per-block results back to host */
    double *h_block = (double *)malloc((size_t)GRID * sizeof(double));
    if (!h_block) {
        fprintf(stderr, "integrate: malloc failed\n");
        cudaFree(d_block);
        return 0.0;
    }

    err = cudaMemcpy(h_block, d_block, (size_t)GRID * sizeof(double),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_block);
    if (err != cudaSuccess) {
        fprintf(stderr, "integrate: cudaMemcpy failed: %s\n",
                cudaGetErrorString(err));
        free(h_block);
        return 0.0;
    }

    /* Sum per-block partials on the host */
    double total = 0.0;
    for (int i = 0; i < GRID; i++) {
        total += h_block[i];
    }
    free(h_block);

    /* Multiply by h to complete the midpoint-rule formula */
    return total * h;
}
