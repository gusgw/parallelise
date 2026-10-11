#include "cg.h"
#include "reference.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void apply_A(double *q, const double *p, long m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * m) return;
    int i = idx / m, j = idx % m;
    double s = 4.0 * p[idx];
    if (i > 0)     s -= p[idx - m];
    if (i < m - 1) s -= p[idx + m];
    if (j > 0)     s -= p[idx - 1];
    if (j < m - 1) s -= p[idx + 1];
    q[idx] = s;
}

__global__ void axpy(double *x, const double *p, double alpha, long n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += alpha * p[idx];
}

__global__ void dot_reduce(double *partial, const double *u, const double *v, long n) {
    extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double sum = u[idx] * v[idx];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sdata[0];
}

void cgsolve(double *x, const double *b, long m, int iters) {
    int n = m * m;
    double *d_x, *d_r, *d_p, *d_q;
    cudaMalloc((void **)&d_x, n * sizeof(double));
    cudaMalloc((void **)&d_r, n * sizeof(double));
    cudaMalloc((void **)&d_p, n * sizeof(double));
    cudaMalloc((void **)&d_q, n * sizeof(double));
    cudaMemcpy(d_r, b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, b, n * sizeof(double), cudaMemcpyHostToDevice);
    double rr = 0.0;
    for (int iter = 0; iter < iters; iter++) {
        apply_A<<<(n + 255) / 256, 256>>>(d_q, d_p, m);
        double alpha = 0.0;
        double *partial = (double *)malloc((n + 255) / 256 * sizeof(double));
        dot_reduce<<<(n + 255) / 256, 256, 256 * sizeof(double)>>>(partial, d_p, d_q, n);
        for (int i = 0; i < (n + 255) / 256; i++) alpha += partial[i];
        double beta = rr / rr;
        axpy<<<(n + 255) / 256, 256>>>(d_p, d_r, 1.0, n);
        axpy<<<(n + 255) / 256, 256>>>(d_p, d_p, beta, n);
        rr = 0.0;
        dot_reduce<<<(n + 255) / 256, 256, 256 * sizeof(double)>>>(partial, d_r, d_r, n);
        for (int i = 0; i < (n + 255) / 256; i++) rr += partial[i];
        free(partial);
        axpy<<<(n + 255) / 256, 256>>>(d_x, d_p, alpha, n);
        axpy<<<(n + 255) / 256, 256>>>(d_r, d_q, -alpha, n);
    }
    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}