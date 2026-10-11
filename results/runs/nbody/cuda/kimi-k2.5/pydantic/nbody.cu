/* nbody.cu - CUDA implementation of direct 2D N-body acceleration */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nbody.h"

#define EPS2 0.0025

/* Grid-stride kernel: each thread computes accelerations for one body i */
__global__ void nbody_kernel(const double *px, const double *py, double *ax, double *ay, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        double sx = 0.0;
        double sy = 0.0;
        double pxi = px[i];
        double pyi = py[i];

        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - pxi;
            double dy = py[j] - pyi;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}

extern "C" void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n)
{
    double *d_px, *d_py, *d_ax, *d_ay;
    size_t size = (size_t)n * sizeof(double);

    /* Allocate device memory */
    cudaError_t err = cudaMalloc((void**)&d_px, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_px: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&d_py, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_py: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&d_ax, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ax: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&d_ay, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ay: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    /* Copy input data to device */
    err = cudaMemcpy(d_px, px, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_px: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(d_py, py, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_py: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    /* Launch kernel with grid-stride pattern */
    int blockSize = 256;
    int numBlocks = (int)((n + blockSize - 1) / blockSize);
    if (numBlocks < 1) numBlocks = 1;
    /* Limit blocks for grid-stride efficiency */
    if (numBlocks > 256) numBlocks = 256;

    nbody_kernel<<<numBlocks, blockSize>>>(d_px, d_py, d_ax, d_ay, n);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    /* Synchronize device */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    /* Copy results back to host */
    err = cudaMemcpy(ax, d_ax, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ax: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(ay, d_ay, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ay: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    /* Free device memory */
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
