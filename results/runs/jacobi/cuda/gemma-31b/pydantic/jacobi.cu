#include <cuda_runtime.h>
#include <stdio.h>
#include "jacobi.h"

__global__ void jacobi_kernel(double *u, double *un, long n) {
    long i = blockIdx.y * blockDim.y + threadIdx.y;
    long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        un[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + 
                                u[i * n + (j - 1)] + u[i * n + (j + 1)]);
    }
}

extern "C" void jacobi(double *u, long n, int iters) {
    double *d_u, *d_un;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&d_u, size);
    cudaMalloc(&d_un, size);

    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_un, u, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    for (int it = 0; it < iters; it++) {
        if (it % 2 == 0) {
            jacobi_kernel<<<grid, block>>>(d_u, d_un, n);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
        } else {
            jacobi_kernel<<<grid, block>>>(d_un, d_u, n);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
        }
    }

    double *final_ptr = (iters % 2 == 0) ? d_u : d_un;
    cudaMemcpy(u, final_ptr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_un);
}
