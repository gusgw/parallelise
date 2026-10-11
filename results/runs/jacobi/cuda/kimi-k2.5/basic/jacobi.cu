#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "jacobi.h"

/* CUDA kernel for one Jacobi sweep: each thread computes one interior cell.
   The grid is 2D with one thread per interior cell. */
__global__ void jacobi_kernel(const double *u_in, double *u_out, long n)
{
    long i = blockIdx.y * blockDim.y + threadIdx.y + 1;  /* row, skip top boundary */
    long j = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* col, skip left boundary */

    if (i < n - 1 && j < n - 1) {
        long idx = i * n + j;
        u_out[idx] = 0.25 * (u_in[(i - 1) * n + j] + u_in[(i + 1) * n + j]
                             + u_in[i * n + j - 1] + u_in[i * n + j + 1]);
    }
}

extern "C" void jacobi(double *u, long n, int iters)
{
    if (n < 2 || iters <= 0) return;

    size_t size = sizeof(double) * (size_t)n * n;

    /* Allocate device buffers */
    double *d_u[2];
    cudaError_t err;

    err = cudaMalloc(&d_u[0], size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_u[1], size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        return;
    }

    /* Copy initial data to device */
    err = cudaMemcpy(d_u[0], u, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);
        return;
    }

    /* Copy boundaries to second buffer as well (they never change) */
    err = cudaMemcpy(d_u[1], u, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);
        return;
    }

    /* Configure kernel: 2D grid for interior cells */
    /* Interior is (n-2) x (n-2) cells */
    dim3 blockDim(16, 16);
    dim3 gridDim((unsigned int)((n - 2 + blockDim.x - 1) / blockDim.x),
                 (unsigned int)((n - 2 + blockDim.y - 1) / blockDim.y));

    int cur = 0;
    int next = 1;

    for (int it = 0; it < iters; it++) {
        jacobi_kernel<<<gridDim, blockDim>>>(d_u[cur], d_u[next], n);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "jacobi_kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u[0]);
            cudaFree(d_u[1]);
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u[0]);
            cudaFree(d_u[1]);
            return;
        }

        /* Swap buffers */
        int tmp = cur;
        cur = next;
        next = tmp;
    }

    /* Copy result back to host */
    err = cudaMemcpy(u, d_u[cur], size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_u[0]);
    cudaFree(d_u[1]);
}
