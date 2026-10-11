#include <cuda_runtime.h>
#include "nbody.h"
#include <math.h>
#include <stdio.h>

__global__ void nbody_accel_kernel(const double *px, const double *py, double *ax, double *ay, long n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sx = 0.0, sy = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = px[j] - px[i], dy = py[j] - py[i];
        double r2 = dx * dx + dy * dy;
        double d3 = (r2 + 0.0025) * sqrt(r2 + 0.0025);
        sx += dx / d3;
        sy += dy / d3;
    }
    ax[i] = sx;
    ay[i] = sy;
}

extern "C" void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n) {
    int num_blocks = (n + 255) / 256;
    int num_threads = 256;
    double *d_px, *d_py, *d_ax, *d_ay;
    cudaMalloc((void **) &d_px, n * sizeof(double));
    cudaMalloc((void **) &d_py, n * sizeof(double));
    cudaMalloc((void **) &d_ax, n * sizeof(double));
    cudaMalloc((void **) &d_ay, n * sizeof(double));
    cudaMemcpy(d_px, px, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py, n * sizeof(double), cudaMemcpyHostToDevice);
    nbody_accel_kernel<<<num_blocks, num_threads>>>(d_px, d_py, d_ax, d_ay, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ax);
    cudaFree(d_ay);
}
