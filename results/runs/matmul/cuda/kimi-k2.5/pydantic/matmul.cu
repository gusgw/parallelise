#include "matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* CUDA kernel for matrix multiplication C = A * B
 * Each thread computes one element of C */
__global__ void matmul_kernel(const double *A, const double *B, double *C, long n)
{
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (long k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

extern "C" void matmul(const double *A, const double *B, double *C, long n)
{
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    size_t size = n * n * sizeof(double);
    cudaError_t err;

    /* Allocate device memory */
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        exit(EXIT_FAILURE);
    }

    /* Copy host memory to device */
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_A failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    /* Define block and grid dimensions */
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    /* Launch kernel */
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul_kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    /* Synchronize device */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    /* Copy result back to host */
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy C failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
