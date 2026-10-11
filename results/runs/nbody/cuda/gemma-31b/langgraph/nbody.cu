#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"

#define EPS2 0.0025

__global__ void nbody_kernel(const double *px, const double *py, double *ax, double *ay, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle n > blockDim.x * gridDim.x
    for (; i < n; i += blockDim.x * gridDim.x) {
        double px_i = px[i];
        double py_i = py[i];
        double sx = 0.0;
        double sy = 0.0;
        
        for (long j = 0; j < n; j++) {
            if (j == i) continue;
            double dx = px[j] - px_i;
            double dy = py[j] - py_i;
            double r2 = dx * dx + dy * dy;
            double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}

extern "C" void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    double *d_px, *d_py, *d_ax, *d_ay;
    
    cudaError_t err = cudaMalloc(&d_px, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc px failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_py, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc py failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_ax, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc ax failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_ay, n * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc ay failed: %s\n", cudaGetErrorString(err)); return; }

    cudaMemcpy(d_px, px, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535; // Cap grid size for safety, grid-stride loop handles the rest

    nbody_kernel<<<gridSize, blockSize>>>(d_px, d_py, d_ax, d_ay, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); }

    cudaMemcpy(ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
