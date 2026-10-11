#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "jacobi.h"

/* CUDA kernel for Jacobi 5-point stencil.
 * Each thread updates one interior cell.
 * Reads from u_cur, writes to u_next.
 */
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

extern "C" void jacobi(double *u, long n, int iters)
{
    if (n < 3 || iters <= 0) {
        return;
    }

    size_t size = sizeof(double) * (size_t)n * n;
    double *d_u[2];
    cudaError_t err;

    /* Allocate two device buffers for double buffering */
    err = cudaMalloc((void **)&d_u[0], size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void **)&d_u[1], size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        return;
    }

    /* Copy initial data to device */
    err = cudaMemcpy(d_u[0], u, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);
        return;
    }

    /* Copy boundaries from u[0] to u[1] (boundaries never change) */
    err = cudaMemcpy(d_u[1], d_u[0], size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);
        return;
    }

    /* Configure kernel launch parameters */
    /* Use 16x16 threads per block for good occupancy */
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    int cur = 0;
    int next = 1;

    /* Run Jacobi iterations */
    for (int it = 0; it < iters; it++) {
        jacobi_kernel<<<gridDim, blockDim>>>(d_u[cur], d_u[next], n);
        
        /* Check for kernel launch errors */
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "jacobi_kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u[0]);
            cudaFree(d_u[1]);
            return;
        }

        /* Synchronize to ensure kernel completion */
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_u[0]);
            cudaFree(d_u[1]);
            return;
        }

        /* Swap buffers for next iteration */
        int tmp = cur;
        cur = next;
        next = tmp;
    }

    /* Copy final result back to host */
    err = cudaMemcpy(u, d_u[cur], size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);
        return;
    }

    /* Free device memory */
    cudaFree(d_u[0]);
    cudaFree(d_u[1]);
}
