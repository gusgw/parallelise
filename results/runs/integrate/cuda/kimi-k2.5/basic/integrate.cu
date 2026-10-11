/* integrate.cu - CUDA implementation of midpoint rule integration
 * Computes integral of 4/(1+x^2) over [a, b] using n subintervals
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integrate.h"

/* Grid-stride kernel: each thread processes multiple elements with stride
 * equal to the total number of threads (gridDim.x * blockDim.x).
 * Uses shared memory for per-block reduction. */
__global__ void integrate_kernel(double a, double b, long n, double *block_sums) {
    extern __shared__ double sdata[];
    
    long tid = threadIdx.x;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;
    
    double h = (b - a) / (double)n;
    double local_sum = 0.0;
    
    /* Grid-stride loop: each thread handles multiple subintervals */
    for (long i = idx; i < n; i += stride) {
        double x = a + ((double)i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    
    /* Load local sum into shared memory */
    sdata[tid] = local_sum;
    __syncthreads();
    
    /* Shared memory reduction within block */
    for (long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    /* Write block result to global memory */
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/* Host function that launches the kernel and returns the integral */
extern "C" double integrate(double a, double b, long n) {
    if (n <= 0) {
        return 0.0;
    }
    
    /* CUDA configuration */
    int threads_per_block = 256;
    int num_blocks = 256;  /* Use multiple blocks for better occupancy */
    
    /* Allocate device memory for block partial sums */
    double *d_block_sums = NULL;
    cudaError_t err = cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }
    
    /* Launch kernel with dynamic shared memory */
    size_t shared_mem_size = threads_per_block * sizeof(double);
    integrate_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        a, b, n, d_block_sums);
    
    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_sums);
        return 0.0;
    }
    
    /* Synchronize and check for execution errors */
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
    
    err = cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double),
                     cudaMemcpyDeviceToHost);
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
    
    /* Multiply by h to get the integral */
    double h = (b - a) / (double)n;
    return total_sum * h;
}
