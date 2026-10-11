#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "integrate.h"

/* CUDA kernel for midpoint-rule integration of 4/(1+x^2) over [a, b]
 * Each thread processes a subset of subintervals using grid-stride loop,
 * then participates in a per-block shared-memory reduction. */

#define BLOCK_SIZE 256

__global__ void integrate_kernel(double a, double b, long n, double *block_sums)
{
    __shared__ double sdata[BLOCK_SIZE];
    
    const double h = (b - a) / (double)n;
    const long tid = blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = blockDim.x * gridDim.x;
    
    double local_sum = 0.0;
    
    /* Grid-stride loop: each thread handles multiple subintervals */
    for (long i = tid; i < n; i += stride) {
        const double x = a + ((double)i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    
    /* Load into shared memory for reduction */
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    
    /* Parallel reduction in shared memory */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    /* Write block result to global memory */
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

extern "C" double integrate(double a, double b, long n)
{
    if (n <= 0) {
        return 0.0;
    }
    
    /* Determine grid size */
    int num_blocks = 0;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }
    
    /* Use a reasonable number of blocks - enough to saturate the GPU */
    /* For large n, we want many blocks; for small n, fewer blocks */
    num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) {
        num_blocks = 1024;
    }
    if (num_blocks < 1) {
        num_blocks = 1;
    }
    
    /* Allocate device memory for block partial sums */
    double *d_block_sums = NULL;
    err = cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }
    
    /* Launch kernel */
    integrate_kernel<<<num_blocks, BLOCK_SIZE>>>(a, b, n, d_block_sums);
    
    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_sums);
        return 0.0;
    }
    
    /* Synchronize device */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_sums);
        return 0.0;
    }
    
    /* Copy block sums back to host */
    double *h_block_sums = (double *)malloc(num_blocks * sizeof(double));
    if (h_block_sums == NULL) {
        fprintf(stderr, "malloc failed\n");
        cudaFree(d_block_sums);
        return 0.0;
    }
    
    err = cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_block_sums);
        cudaFree(d_block_sums);
        return 0.0;
    }
    
    /* Sum up block results */
    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_block_sums[i];
    }
    
    /* Multiply by h to get the integral */
    const double h = (b - a) / (double)n;
    double result = total_sum * h;
    
    /* Cleanup */
    free(h_block_sums);
    cudaFree(d_block_sums);
    
    return result;
}
