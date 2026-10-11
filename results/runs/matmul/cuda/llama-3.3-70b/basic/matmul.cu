#include <cuda_runtime.h>
#include "matmul.h"
#include <math.h>
#include <stdio.h>

__global__ void matmul_kernel(const double *A, const double *B, double *C, long n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = sum;
    }
}

extern "C" void matmul(const double *A, const double *B, double *C, long n) {
    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(double);
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
