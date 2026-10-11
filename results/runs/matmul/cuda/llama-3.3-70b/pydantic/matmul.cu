// matmul.cu: parallel matrix multiply kernel
#include "matmul.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h> // include stdio.h for printf
__global__ void matmul_kernel(const double *A, const double *B, double *C, long n) {
    // each thread computes one entry of C
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
    // allocate device buffers
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * n * sizeof(double));
    cudaMalloc((void **)&d_B, n * n * sizeof(double));
    cudaMalloc((void **)&d_C, n * n * sizeof(double));

    // copy A and B to device
    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // launch kernel
    int numBlocks = (n + 15) / 16;
    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(16, 16);
    matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // synchronise and check CUDA errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // copy C back to host
    cudaMemcpy(C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // free device buffers
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}