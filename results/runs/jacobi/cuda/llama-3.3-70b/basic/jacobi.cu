#include "jacobi.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void jacobi_kernel(double *u, double *un, long n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        int idx = i * n + j;
        un[idx] = 0.25 * (u[idx - n] + u[idx + n] + u[idx - 1] + u[idx + 1]);
    } else if (i == 0) {
        int idx = i * n + j;
        un[idx] = 1.0;
    } else if (i == n - 1) {
        int idx = i * n + j;
        un[idx] = 0.0;
    } else if (j == 0) {
        int idx = i * n + j;
        un[idx] = 0.0;
    } else if (j == n - 1) {
        int idx = i * n + j;
        un[idx] = 0.0;
    }
}

void jacobi(double *u, long n, int iters) {
    double *d_u, *d_un;
    cudaMalloc((void **)&d_u, n * n * sizeof(double));
    cudaMalloc((void **)&d_un, n * n * sizeof(double));
    cudaMemcpy(d_u, u, n * n * sizeof(double), cudaMemcpyHostToDevice);
    for (int iter = 0; iter < iters; iter++) {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);
        jacobi_kernel<<<dimGrid, dimBlock>>>(d_u, d_un, n);
        cudaDeviceSynchronize();
        double *tmp = d_u;
        d_u = d_un;
        d_un = tmp;
    }
    cudaMemcpy(u, d_u, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    cudaFree(d_un);
}