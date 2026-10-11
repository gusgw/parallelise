/* nbody.cu - CUDA implementation of direct 2D N-body acceleration */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"

#define EPS2 0.0025
#define BLOCK_SIZE 256

/* CUDA kernel: each thread computes acceleration for one body i */
__global__ void nbody_kernel(const double *px, const double *py,
                             double *ax, double *ay, long n)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    const double xi = px[i];
    const double yi = py[i];
    double sx = 0.0;
    double sy = 0.0;
    
    for (long j = 0; j < n; j++) {
        if (j == i) continue;
        const double dx = px[j] - xi;
        const double dy = py[j] - yi;
        const double r2 = dx * dx + dy * dy;
        const double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
        sx += dx / d3;
        sy += dy / d3;
    }
    
    ax[i] = sx;
    ay[i] = sy;
}

/* Host function: copies data to device, launches kernel, copies result back */
extern "C" void nbody_accel(const double *px, const double *py,
                            double *ax, double *ay, long n)
{
    double *d_px = NULL;
    double *d_py = NULL;
    double *d_ax = NULL;
    double *d_ay = NULL;
    cudaError_t err;
    size_t size;
    long num_blocks;
    
    size = sizeof(double) * n;
    num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    /* Allocate device memory */
    err = cudaMalloc(&d_px, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_px: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_py, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_py: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        return;
    }
    
    err = cudaMalloc(&d_ax, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ax: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        cudaFree(d_py);
        return;
    }
    
    err = cudaMalloc(&d_ay, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ay: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        cudaFree(d_py);
        cudaFree(d_ax);
        return;
    }
    
    /* Copy input data to device */
    err = cudaMemcpy(d_px, px, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_px: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(d_py, py, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_py: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    /* Launch kernel */
    nbody_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_ax, d_ay, n);
    
    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    /* Synchronize device */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    /* Copy results back to host */
    err = cudaMemcpy(ax, d_ax, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ax: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(ay, d_ay, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ay: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
cleanup:
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
