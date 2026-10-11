// integrate.cu: midpoint-rule estimate of the integral of 4/(1+x^2) over [a, b] using n subintervals
#include "integrate.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
__global__ void integrate_kernel(double *terms, double a, double b, long n) {
    // each thread accumulates its share of the subintervals
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a + (idx + 0.5) * (b - a) / n;
        terms[idx] = 4.0 / (1.0 + x * x) * (b - a) / n;
    }
}

// host function with the exact signature from the header
extern "C" double integrate(double a, double b, long n) {
    // launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    double *terms, *host_terms;
    cudaMalloc((void **)&terms, n * sizeof(double));
    host_terms = (double *)malloc(n * sizeof(double));
    integrate_kernel<<<numBlocks, blockSize>>>(terms, a, b, n);
    // wait and check CUDA API and kernel errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    // copy terms from device to host
    cudaMemcpy(host_terms, terms, n * sizeof(double), cudaMemcpyDeviceToHost);
    // combine the per-block partial sums
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += host_terms[i];
    }
    cudaFree(terms);
    free(host_terms);
    return sum;
}
