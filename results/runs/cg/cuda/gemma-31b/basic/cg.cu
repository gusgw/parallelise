#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

#define CHECK_CUDA(call)                                                               \
    do {                                                                              \
        cudaError_t err = call;                                                       \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err));                                          \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

extern "C" void cgsolve(double *x, const double *b, long m, int iters) {
    long n = m * m;
    double *d_x, *d_b, *d_r, *d_p, *d_q;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_q, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double *d_partials;
    CHECK_CUDA(cudaMalloc(&d_partials, blocksPerGrid * sizeof(double)));

    auto get_dot = [&](double *u, double *v) {
        CHECK_CUDA(cudaMemset(d_partials, 0, blocksPerGrid * sizeof(double)));
        dot_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(u, v, d_partials, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        double h_partials[blocksPerGrid];
        CHECK_CUDA(cudaMemcpy(h_partials, d_partials, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
        double res = 0.0;
        for (int i = 0; i < blocksPerGrid; i++) res += h_partials[i];
        return res;
    };

    double rr = get_dot(d_r, d_r);

    for (int it = 0; it < iters; it++) {
        apply_A_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_q, d_p, m);
        double pq = get_dot(d_p, d_q);
        double alpha = rr / pq;

        axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_p, alpha, n, 1);
        axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_q, alpha, n, -1);

        double rr_new = get_dot(d_r, d_r);
        double beta = rr_new / rr;
        rr = rr_new;

        p_update_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_p, d_r, beta, n);
    }

    CHECK_CUDA(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_partials));
}
