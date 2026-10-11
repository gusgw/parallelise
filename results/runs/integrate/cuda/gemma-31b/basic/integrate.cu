#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "integrate.h"

__global__ void integrate_kernel(double a, double b, long n, double h, double* partial_sums) {
    extern __shared__ double sdata[];

    double local_sum = 0.0;
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = tid; i < n; i += stride) {
        double x = a + (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    sdata[threadIdx.x] = local_sum;
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

extern "C" double integrate(double a, double b, long n) {
    double h = (b - a) / n;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535; // Cap for sanity, grid-stride handles the rest

    double *d_partial_sums;
    cudaMalloc(&d_partial_sums, blocksPerGrid * sizeof(double));

    integrate_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(a, b, n, h, d_partial_sums);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_partial_sums);
        return 0.0;
    }

    double *h_partial_sums = (double *)malloc(blocksPerGrid * sizeof(double));
    cudaMemcpy(h_partial_sums, d_partial_sums, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    double total_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += h_partial_sums[i];
    }

    cudaFree(d_partial_sums);
    free(h_partial_sums);

    return total_sum * h;
}
