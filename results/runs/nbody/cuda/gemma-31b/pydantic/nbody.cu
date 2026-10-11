#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "nbody.h"

#define EPS2 0.0025

__global__ void nbody_kernel(const double *px, const double *py, double *ax, double *ay, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

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

extern "C" void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    double *d_px, *d_py, *d_ax, *d_ay;
    size_t size_px = n * sizeof(double);

    cudaMalloc(&d_px, size_px);
    cudaMalloc(&d_py, size_px);
    cudaMalloc(&d_ax, size_px);
    cudaMalloc(&d_ay, size_px);

    cudaMemcpy(d_px, px, size_px, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, size_px, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    nbody_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_px, d_py, d_ax, d_ay, n);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(ax, d_ax, size_px, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, d_ay, size_px, cudaMemcpyDeviceToHost);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
