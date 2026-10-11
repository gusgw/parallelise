#include "cg.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void cg_kernel(float *x, float *b, int m, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * m) return;
    float sum = 0.0f;
    for (int i = 0; i < iters; i++) {
        float val = b[idx];
        if (idx >= m) val -= x[idx - m];
        if (idx < m * m - m) val -= x[idx + m];
        if (idx % m > 0) val -= x[idx - 1];
        if (idx % m < m - 1) val -= x[idx + 1];
        sum += val;
    }
    x[idx] = sum;
}

float cg(float *x, float *b, int m, int iters) {
    float *d_x, *d_b;
    cudaMalloc((void **)&d_x, m * m * sizeof(float));
    cudaMalloc((void **)&d_b, m * m * sizeof(float));
    cudaMemcpy(d_b, b, m * m * sizeof(float), cudaMemcpyHostToDevice);
    cg_kernel<<<(m * m + 255) / 256, 256>>>(d_x, d_b, m, iters);
    cudaDeviceSynchronize();
    float checksum = 0.0f;
    cudaMemcpy(x, d_x, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m * m; i++) checksum += x[i];
    cudaFree(d_x);
    cudaFree(d_b);
    return checksum;
}