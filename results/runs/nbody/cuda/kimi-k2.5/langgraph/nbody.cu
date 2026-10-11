#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"

#define EPS2 0.0025

/* CUDA kernel: each thread computes the acceleration for one body i.
 * Uses grid-stride loop to handle any n. */
__global__ void nbody_kernel(const double *px, const double *py,
                             double *ax, double *ay, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    
    for (; i < n; i += (long)blockDim.x * gridDim.x) {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0;
        double sy = 0.0;
        
        for (long j = 0; j < n; j++) {
            if (j == i)
                continue;
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
}

extern "C" void nbody_accel(const double *px, const double *py,
                            double *ax, double *ay, long n)
{
    double *d_px = NULL;
    double *d_py = NULL;
    double *d_ax = NULL;
    double *d_ay = NULL;
    
    size_t nbytes = sizeof(double) * (size_t)n;
    
    cudaError_t err;
    
    /* Launch configuration: use 256 threads per block, enough blocks to cover n */
    const int threadsPerBlock = 256;
    int blocks = (int)((n + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;  /* reasonable upper limit */
    
    err = cudaMalloc((void **)&d_px, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_px: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    err = cudaMalloc((void **)&d_py, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_py: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        exit(1);
    }
    
    err = cudaMalloc((void **)&d_ax, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ax: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        cudaFree(d_py);
        exit(1);
    }
    
    err = cudaMalloc((void **)&d_ay, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_ay: %s\n", cudaGetErrorString(err));
        cudaFree(d_px);
        cudaFree(d_py);
        cudaFree(d_ax);
        exit(1);
    }
    
    err = cudaMemcpy(d_px, px, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_px: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(d_py, py, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_py: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    nbody_kernel<<<blocks, threadsPerBlock>>>(d_px, d_py, d_ax, d_ay, n);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(ax, d_ax, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ax: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpy(ay, d_ay, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for ay: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
cleanup:
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
    
    if (err != cudaSuccess) {
        exit(1);
    }
}
