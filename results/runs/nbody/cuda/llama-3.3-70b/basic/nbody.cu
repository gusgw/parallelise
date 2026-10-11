#include <cuda_runtime.h>
#include <cuda.h>
#include "nbody.h"
#include <math.h>
#include <stdio.h>
#define EPS2 0.0025
#define N_TEST 1024
__global__ void nbody_accel_kernel(const double *px, const double *py, double *ax, double *ay, long n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sx = 0.0;
    double sy = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = px[j] - px[i];
        double dy = py[j] - py[i];
        double r2 = dx * dx + dy * dy;
        double d3 = (r2 + EPS2) * sqrt(r2 + EPS2);
        sx += dx / d3;
        sy += dy / d3;
    }
    ax[i] = sx;
    ay[i] = sy;
}

void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    double *d_px, *d_py, *d_ax, *d_ay;
    cudaMalloc((void **)&d_px, n * sizeof(double));
    cudaMalloc((void **)&d_py, n * sizeof(double));
    cudaMalloc((void **)&d_ax, n * sizeof(double));
    cudaMalloc((void **)&d_ay, n * sizeof(double));
    cudaMemcpy(d_px, px, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, n * sizeof(double), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    nbody_accel_kernel<<<numBlocks, blockSize>>>(d_px, d_py, d_ax, d_ay, n);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    cudaMemcpy(ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
