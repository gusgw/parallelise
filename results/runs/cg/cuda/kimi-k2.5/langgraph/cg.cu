/* cg.cu - CUDA implementation of conjugate gradient for 2D Poisson system */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cg.h"

/* Kernel: apply 5-point Poisson stencil (q = A * p) */
__global__ void apply_A_kernel(const double *p, double *q, long m)
{
    long n = m * m;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (long k = idx; k < n; k += blockDim.x * gridDim.x) {
        long i = k / m;
        long j = k % m;
        double s = 4.0 * p[k];
        if (i > 0)     s -= p[k - m];
        if (i < m - 1) s -= p[k + m];
        if (j > 0)     s -= p[k - 1];
        if (j < m - 1) s -= p[k + 1];
        q[k] = s;
    }
}

/* Kernel: x += alpha * p; r -= alpha * q (combined axpy) */
__global__ void update_xr_kernel(double *x, double *r, const double *p, 
                                  const double *q, double alpha, long n)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long k = idx; k < n; k += blockDim.x * gridDim.x) {
        x[k] += alpha * p[k];
        r[k] -= alpha * q[k];
    }
}

/* Kernel: p = r + beta * p */
__global__ void update_p_kernel(double *p, const double *r, double beta, long n)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (long k = idx; k < n; k += blockDim.x * gridDim.x) {
        p[k] = r[k] + beta * p[k];
    }
}

/* Kernel: dot product with per-block reduction using shared memory */
__global__ void dot_kernel(const double *u, const double *v, double *block_sums, long n)
{
    extern __shared__ double sdata[];
    
    long tid = threadIdx.x;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    for (long k = idx; k < n; k += blockDim.x * gridDim.x) {
        sum += u[k] * v[k];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/* Host function: compute dot product on GPU */
static double gpu_dot(const double *d_u, const double *d_v, double *d_block_sums,
                      double *h_block_sums, int num_blocks, int block_size, long n)
{
    dot_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
        d_u, d_v, d_block_sums, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "dot_kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaDeviceSynchronize();
    
    err = cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), 
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    double sum = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        sum += h_block_sums[i];
    }
    return sum;
}

extern "C" void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;
    
    // Allocate device memory
    double *d_x, *d_r, *d_p, *d_q, *d_b;
    cudaError_t err;
    
    err = cudaMalloc(&d_x, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    err = cudaMalloc(&d_r, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    err = cudaMalloc(&d_p, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    err = cudaMalloc(&d_q, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    err = cudaMalloc(&d_b, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    
    // Copy b to device
    err = cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed\n"); exit(1); }
    
    // Initialize: x = 0, r = b, p = r
    err = cudaMemset(d_x, 0, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemset failed\n"); exit(1); }
    err = cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed\n"); exit(1); }
    err = cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed\n"); exit(1); }
    
    // Setup for dot product reduction
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > 1024) num_blocks = 1024;  // Limit blocks for reduction
    
    double *d_block_sums;
    err = cudaMalloc(&d_block_sums, num_blocks * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    
    double *h_block_sums = (double *)malloc(num_blocks * sizeof(double));
    if (!h_block_sums) { fprintf(stderr, "malloc failed\n"); exit(1); }
    
    // rr = dot(r, r)
    double rr = gpu_dot(d_r, d_r, d_block_sums, h_block_sums, num_blocks, block_size, n);
    
    // CG iterations
    for (int it = 0; it < iters; it++) {
        // q = A * p
        apply_A_kernel<<<num_blocks, block_size>>>(d_p, d_q, m);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "apply_A_kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaDeviceSynchronize();
        
        // dot(p, q)
        double pdotq = gpu_dot(d_p, d_q, d_block_sums, h_block_sums, num_blocks, block_size, n);
        
        // alpha = rr / dot(p, q)
        double alpha = rr / pdotq;
        
        // x += alpha * p; r -= alpha * q
        update_xr_kernel<<<num_blocks, block_size>>>(d_x, d_r, d_p, d_q, alpha, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "update_xr_kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaDeviceSynchronize();
        
        // rr_new = dot(r, r)
        double rr_new = gpu_dot(d_r, d_r, d_block_sums, h_block_sums, num_blocks, block_size, n);
        
        // beta = rr_new / rr
        double beta = rr_new / rr;
        rr = rr_new;
        
        // p = r + beta * p
        update_p_kernel<<<num_blocks, block_size>>>(d_p, d_r, beta, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "update_p_kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaDeviceSynchronize();
    }
    
    // Copy result back to host
    err = cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // Cleanup
    free(h_block_sums);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_block_sums);
}
