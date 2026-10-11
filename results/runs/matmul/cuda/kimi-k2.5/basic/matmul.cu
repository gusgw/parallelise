#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "matmul.h"

/* CUDA kernel for matrix multiplication C = A * B
 * Each thread computes one element of C using a grid-stride pattern */
__global__ void matmul_kernel(const double *A, const double *B, double *C, long n)
{
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Grid-stride loop to handle matrices larger than the grid */
    for (long i = row; i < n; i += blockDim.y * gridDim.y) {
        for (long j = col; j < n; j += blockDim.x * gridDim.x) {
            double sum = 0.0;
            for (long k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

extern "C" void matmul(const double *A, const double *B, double *C, long n)
{
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    size_t size = sizeof(double) * n * n;
    cudaError_t err;
    dim3 blockSize;
    dim3 gridSize;

    /* Allocate device memory */
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for A: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    /* Copy input matrices from host to device */
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for A: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for B: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    /* Configure kernel launch parameters */
    blockSize = dim3(16, 16);
    gridSize = dim3((n + blockSize.x - 1) / blockSize.x, 
                    (n + blockSize.y - 1) / blockSize.y);
    
    /* Launch kernel */
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    /* Synchronize device */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    /* Copy result matrix from device to host */
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for C: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

cleanup:
    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
