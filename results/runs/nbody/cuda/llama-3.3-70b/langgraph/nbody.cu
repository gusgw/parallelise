#include "nbody.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
extern "C" {
__global__ void nbody_kernel(float *px, float *py, float *ax, float *ay, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sx = 0.0;
    float sy = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float dx = px[j] - px[i];
        float dy = py[j] - py[i];
        float r2 = dx * dx + dy * dy;
        float d3 = (r2 + 0.0025) * sqrt(r2 + 0.0025);
        sx += dx / d3;
        sy += dy / d3;
    }
    ax[i] = sx;
    ay[i] = sy;
}

void nbody(float *px, float *py, float *ax, float *ay, int n) {
    int numBlocks = (n + 255) / 256;
    int numThreads = 256;
    nbody_kernel<<<numBlocks, numThreads>>>(px, py, ax, ay, n);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

void nbody_accel(double *px, double *py, double *ax, double *ay, int n) {
    float *px_f = (float *)malloc(n * sizeof(float));
    float *py_f = (float *)malloc(n * sizeof(float));
    float *ax_f = (float *)malloc(n * sizeof(float));
    float *ay_f = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        px_f[i] = (float)px[i];
        py_f[i] = (float)py[i];
    }
    nbody(px_f, py_f, ax_f, ay_f, n);
    for (int i = 0; i < n; i++) {
        ax[i] = (double)ax_f[i];
        ay[i] = (double)ay_f[i];
    }
    free(px_f);
    free(py_f);
    free(ax_f);
    free(ay_f);
}
}