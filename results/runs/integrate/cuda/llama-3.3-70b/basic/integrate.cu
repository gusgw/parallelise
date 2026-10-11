#include "integrate.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void integrate_kernel(double *result, double a, double b, long n) {
    // each thread accumulates its share of the subintervals
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a + (idx + 0.5) * (b - a) / n;
        result[idx] = 4.0 / (1.0 + x * x) * (b - a) / n;
    }
}

double integrate(double a, double b, long n) {
    double *d_result;
    cudaMalloc((void **) &d_result, n * sizeof(double));
    integrate_kernel<<<(n + 255) / 256, 256>>>(d_result, a, b, n);
    cudaDeviceSynchronize();
    double result = 0.0;
    double *h_result = (double *) malloc(n * sizeof(double));
    cudaMemcpy(h_result, d_result, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        result += h_result[i];
    }
    free(h_result);
    cudaFree(d_result);
    return result;
}
