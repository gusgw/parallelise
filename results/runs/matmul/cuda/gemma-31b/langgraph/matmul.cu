#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"

__global__ void matmul_kernel(const double *A, const double *B, double *C, long n) {
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

extern "C" void matmul(const double *A, const double *B, double *C, long n) {
    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(double);

    cudaError_t err;
    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc A failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc C failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy A failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy B failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy C failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
