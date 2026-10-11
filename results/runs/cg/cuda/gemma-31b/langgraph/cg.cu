#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cg.h"

#define BLOCK_SIZE 256

__global__ void apply_A_kernel(double *q, const double *p, long m) {
    long n = m * m;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * p[idx];
        if (i > 0)     s -= p[idx - m];
        if (i < m - 1) s -= p[idx + m];
        if (j > 0)     s -= p[idx - 1];
        if (j < m - 1) s -= p[idx + 1];
        q[idx] = s;
    }
}

__global__ void axpy_kernel(double *x, const double *y, double alpha, long n, int sign) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (sign > 0) x[idx] += alpha * y[idx];
        else           x[idx] -= alpha * y[idx];
    }
}

__global__ void p_update_kernel(double *p, const double *r, double beta, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = r[idx] + beta * p[idx];
    }
}

__global__ void dot_kernel(const double *u, const double *v, double *partial_sums, long n) {
    __shared__ double s_mem[BLOCK_SIZE];
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    while (idx < n) {
        sum += u[idx] * v[idx];
        idx += blockDim.x * gridDim.x;
    }
    s_mem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = s_mem[0];
    }
}

extern "C" void cgsolve(double *x, const double *b, long m, int iters) {
    long n = m * m;
    size_t size = n * sizeof(double);

    double *d_x, *d_b, *d_r, *d_p, *d_q;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_r, size);
    cudaMalloc(&d_p, size);
    cudaMalloc(&d_q, size);

    cudaMemset(d_x, 0, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, d_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, d_r, size, cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *d_partials;
    cudaMalloc(&d_partials, blocks * sizeof(double));
    double *h_partials = (double *)malloc(blocks * sizeof(double));

    auto get_dot = [&](double *u, double *v) {
        cudaMemset(d_partials, 0, blocks * sizeof(double));
        dot_kernel<<<blocks, BLOCK_SIZE>>>(u, v, d_partials, n);
        cudaDeviceSynchronize();
        cudaMemcpy(h_partials, d_partials, blocks * sizeof(double), cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (int i = 0; i < blocks; i++) sum += h_partials[i];
        return sum;
    };

    double rr = get_dot(d_r, d_r);

    for (int iter = 0; iter < iters; iter++) {
        apply_A_kernel<<<blocks, BLOCK_SIZE>>>(d_q, d_p, m);
        double pq = get_dot(d_p, d_q);
        double alpha = rr / pq;

        axpy_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_p, alpha, n, 1);
        axpy_kernel<<<blocks, BLOCK_SIZE>>>(d_r, d_q, alpha, n, -1);

        double rr_new = get_dot(d_r, d_r);
        double beta = rr_new / rr;
        rr = rr_new;

        p_update_kernel<<<blocks, BLOCK_SIZE>>>(d_p, d_r, beta, n);
    }

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_partials);
    free(h_partials);
}
