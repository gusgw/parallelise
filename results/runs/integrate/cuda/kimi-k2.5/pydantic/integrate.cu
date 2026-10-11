#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "integrate.h"

#define BLOCK_SIZE 256

/* CUDA kernel: grid-stride loop for midpoint rule integration.
 * Each thread accumulates its share of subintervals, then
 * participates in a shared-memory reduction within the block. */
__global__ void integrate_kernel(double a, double b, long n, double *block_sums) {
    __shared__ double sdata[BLOCK_SIZE];
    
    double h = (b - a) / n;
    double sum = 0.0;
    
    /* Grid-stride loop: each thread handles multiple elements */
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    
    for (long i = idx; i < n; i += stride) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    
    /* Store thread's partial sum in shared memory */
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    /* Shared-memory reduction within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    /* Write block's sum to global memory */
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/* Host function: launches CUDA kernel and returns the integral result.
 * Implements the exact signature from integrate.h */
extern "C"
double integrate(double a, double b, long n) {
    if (n <= 0) {
        return 0.0;
    }
    
    /* Calculate grid dimensions */
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) {
        num_blocks = 1024;  /* Cap at reasonable number for reduction */
    }
    if (num_blocks < 1) {
        num_blocks = 1;
    }
    
    /* Allocate device memory for block partial sums */
    double *d_block_sums = NULL;
    cudaError_t err = cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
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
    if (!h_block_sums) {
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
    
    /* Sum up block results on host */
    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_block_sums[i];
    }
    
    /* Clean up */
    free(h_block_sums);
    cudaFree(d_block_sums);
    
    /* Multiply by h to get final integral value */
    double h = (b - a) / n;
    return total_sum * h;
}
