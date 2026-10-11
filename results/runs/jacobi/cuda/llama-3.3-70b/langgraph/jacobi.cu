// jacobi.cu: CUDA kernel for 2D Jacobi heat stencil
#include "jacobi.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void jacobi_kernel(double *u, double *u_new, long n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;  // out of bounds
    if (i == 0 || j == 0 || i == n - 1 || j == n - 1) return;  // boundary
    for (int iter = 0; iter < iters; iter++) {
        u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
        __syncthreads();
        u[i * n + j] = u_new[i * n + j];
    }
}

extern "C" void jacobi(double *u, long n, int iters) {
    double *d_u, *d_u_new;
    cudaMalloc((void **)&d_u, n * n * sizeof(double));
    cudaMalloc((void **)&d_u_new, n * n * sizeof(double));
    cudaMemcpy(d_u, u, n * n * sizeof(double), cudaMemcpyHostToDevice);
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    jacobi_kernel<<<grid, block>>>(d_u, d_u_new, n, iters);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    cudaMemcpy(u, d_u, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    cudaFree(d_u_new);
}