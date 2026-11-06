#include <cstdio>
#include <cstdlib>

#include "matmul.h"

/* Reference solution (validates the harness -- NOT shown to the agents). One thread per
 * entry of C in a 2D grid; the host wrapper owns the device buffers and the copies. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ static void matmul_kernel(const double *A, const double *B, double *C, long n)
{
    const long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n)
        return;
    double s = 0.0;
    for (long k = 0; k < n; k++)
        s += A[i * n + k] * B[k * n + j];
    C[i * n + j] = s;
}

extern "C" void matmul(const double *A, const double *B, double *C, long n)
{
    const size_t bytes = sizeof(double) * (size_t)n * n;
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + 15) / 16), (unsigned)((n + 15) / 16));
    matmul_kernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
