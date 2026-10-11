#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "jacobi.h"

/* CUDA kernel for one Jacobi sweep: each thread updates one interior cell.
 * Reads from u_cur, writes to u_next. Boundaries are not modified. */
__global__ void jacobi_kernel(const double *u_cur, double *u_next, long n)
{
    long i = blockIdx.y * blockDim.y + threadIdx.y;
    long j = blockIdx.x * blockDim.x + threadIdx.x;

    /* Only process interior cells (1 to n-2) */
    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1) {
        long idx = i * n + j;
        u_next[idx] = 0.25 * (u_cur[(i - 1) * n + j] +
                              u_cur[(i + 1) * n + j] +
                              u_cur[i * n + j - 1] +
                              u_cur[i * n + j + 1]);
    }
}

/* Host function: copies data to GPU, runs Jacobi iterations, copies result back.
 * Uses double-buffering on the device. */
extern "C" void jacobi(double *u, long n, int iters)
{
    if (n < 2 || iters <= 0) return;

    size_t size = sizeof(double) * (size_t)n * n;

    /* Allocate device buffers */
    double *d_u0 = NULL;
    double *d_u1 = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_u0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_u1, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u0);
        return;
    }

    /* Copy initial data to device */
    err = cudaMemcpy(d_u0, u, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u0);
        cudaFree(d_u1);
        return;
    }

    /* Copy boundaries from d_u0 to d_u1 (boundaries are fixed) */
    err = cudaMemcpy(d_u1, d_u0, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u0);
        cudaFree(d_u1);
        return;
    }

    /* Configure kernel launch parameters */
    /* Use 16x16 threads per block for good occupancy */
    dim3 blockDim(16, 16);
    /* Grid covers the entire n x n grid */
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    double *cur = d_u0;
    double *next = d_u1;

    /* Run iterations */
    for (int it = 0; it < iters; it++) {
        jacobi_kernel<<<gridDim, blockDim>>>(cur, next, n);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "jacobi_kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u0);
            cudaFree(d_u1);
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u0);
            cudaFree(d_u1);
            return;
        }

        /* Swap buffers for next iteration */
        double *tmp = cur;
        cur = next;
        next = tmp;
    }

    /* Copy result back to host (cur holds the final result) */
    err = cudaMemcpy(u, cur, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u0);
        cudaFree(d_u1);
        return;
    }

    /* Free device memory */
    cudaFree(d_u0);
    cudaFree(d_u1);
}
