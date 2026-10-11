#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matmul.h"

/* CUDA kernel: each thread computes one element of C = A * B
 * A, B, C are n x n row-major matrices in device memory */
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

/* Host function: copies A and B to device, launches kernel, copies C back */
extern "C" void matmul(const double *A, const double *B, double *C, long n)
{
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    size_t size = sizeof(double) * n * n;
    cudaError_t err;
    int alloc_failed = 0;

    /* Allocate device memory */
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        alloc_failed = 1;
    }

    if (!alloc_failed) {
        err = cudaMalloc((void **)&d_B, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
            alloc_failed = 1;
        }
    }

    if (!alloc_failed) {
        err = cudaMalloc((void **)&d_C, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
            alloc_failed = 1;
        }
    }

    if (alloc_failed) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    /* Copy A and B from host to device */
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_A failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    /* Launch kernel with 2D grid */
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "matmul_kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    /* Synchronize and check for execution errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    /* Copy C from device to host */
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy C failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
