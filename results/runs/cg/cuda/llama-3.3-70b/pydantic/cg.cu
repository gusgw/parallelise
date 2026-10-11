// cg.cu: CUDA kernel for the conjugate gradient algorithm
#include <cuda_runtime.h>
#include "cg.h"
#include <math.h>
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

__global__ void dot_product(double *partial_sums, const double *u, const double *v, long n) {
    __shared__ double cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double sum = u[idx] * v[idx];
    cache[threadIdx.x] = sum;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = cache[0];
}

double dot_product(double *u, double *v, long n) {
    double *partial_sums;
    cudaMalloc((void **)&partial_sums, ((n + 255) / 256) * sizeof(double));
    dot_product<<<(n + 255) / 256, 256>>>(partial_sums, u, v, n);
    double sum = 0.0;
    cudaMemcpy(&sum, partial_sums, sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 1; i < (n + 255) / 256; i++) {
        double tmp;
        cudaMemcpy(&tmp, partial_sums + i, sizeof(double), cudaMemcpyDeviceToHost);
        sum += tmp;
    }
    cudaFree(partial_sums);
    return sum;
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
    double rr = dot_product(d_r, d_r, n);
    for (int iter = 0; iter < iters; iter++) {
        apply_A<<<(n + 255) / 256, 256>>>(d_q, d_p, m);
        double alpha = rr / dot_product(d_p, d_q, n);
        axpy<<<(n + 255) / 256, 256>>>(d_x, d_p, alpha, n);
        axpy<<<(n + 255) / 256, 256>>>(d_r, d_q, -alpha, n);
        rr = dot_product(d_r, d_r, n);
        axpy<<<(n + 255) / 256, 256>>>(d_p, d_r, 1.0, n);
        axpy<<<(n + 255) / 256, 256>>>(d_p, d_p, rr / rr, n);
    }
    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}