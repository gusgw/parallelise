#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
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
    size_t size = n * sizeof(double);

    cudaError_t err;
    err = cudaMalloc(&d_px, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc px failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_py, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc py failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_ax, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc ax failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_ay, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc ay failed: %s\n", cudaGetErrorString(err)); return; }

    err = cudaMemcpy(d_px, px, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy px failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy(d_py, py, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy py failed: %s\n", cudaGetErrorString(err)); return; }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    nbody_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_px, d_py, d_ax, d_ay, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "cudaGetLastError: %s\n", cudaGetErrorString(err)); return; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); return; }

    err = cudaMemcpy(ax, d_ax, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy ax failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy(ay, d_ay, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy ay failed: %s\n", cudaGetErrorString(err)); return; }

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
