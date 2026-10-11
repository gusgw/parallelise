#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cg.h"

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
    extern __shared__ double sdata[];
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    while (idx < n) {
        sum += u[idx] * v[idx];
        idx += blockDim.x * gridDim.x;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

extern "C" void cgsolve(double *x, const double *b, long m, int iters) {
    long n = m * m;
    double *d_x, *d_b, *d_r, *d_p, *d_q;
    check_cuda(cudaMalloc(&d_x, n * sizeof(double)));
    check_cuda(cudaMalloc(&d_b, n * sizeof(double)));
    check_cuda(cudaMalloc(&d_r, n * sizeof(double)));
    check_cuda(cudaMalloc(&d_p, n * sizeof(double)));
    check_cuda(cudaMalloc(&d_q, n * sizeof(double)));

    check_cuda(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_x, 0, n * sizeof(double)));
    check_cuda(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    check_cuda(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    double *d_partials;
    check_cuda(cudaMalloc(&d_partials, blocks * sizeof(double)));
    double *h_partials = (double *)malloc(blocks * sizeof(double));

    auto get_dot = [&](double *u, double *v) {
        check_cuda(cudaMemset(d_partials, 0, blocks * sizeof(double)));
        dot_kernel<<<blocks, threads, threads * sizeof(double)>>>(u, v, d_partials, n);
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(h_partials, d_partials, blocks * sizeof(double), cudaMemcpyDeviceToHost));
        double res = 0.0;
        for (int i = 0; i < blocks; i++) res += h_partials[i];
        return res;
    };

    double rr = get_dot(d_r, d_r);

    for (int it = 0; it < iters; it++) {
        apply_A_kernel<<<blocks, threads>>>(d_q, d_p, m);
        double pq = get_dot(d_p, d_q);
        double alpha = rr / pq;

        axpy_kernel<<<blocks, threads>>>(d_x, d_p, alpha, n, 1);
        axpy_kernel<<<blocks, threads>>>(d_r, d_q, alpha, n, -1);

        double rr_new = get_dot(d_r, d_r);
        double beta = rr_new / rr;
        rr = rr_new;

        p_update_kernel<<<blocks, threads>>>(d_p, d_r, beta, n);
    }

    check_cuda(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    free(h_partials);
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_b));
    check_cuda(cudaFree(d_r));
    check_cuda(cudaFree(d_p));
    check_cuda(cudaFree(d_q));
    check_cuda(cudaFree(d_partials));
}
