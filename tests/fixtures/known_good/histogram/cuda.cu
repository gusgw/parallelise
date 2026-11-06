#include <cstdio>
#include <cstdlib>

#include "histogram.h"

/* Known-good routine (validates the harness; never shown to an agent). The threads stride
 * through the values together, and each adds one to its value's bin with an atomic addition
 * in the graphics card's memory. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCKS = 1024, THREADS = 256;

__global__ static void tally(const double *x, long n, int nbins, unsigned long long *counts)
{
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        long b = (long)(x[i] * nbins);
        if (b > nbins - 1)
            b = nbins - 1;
        atomicAdd(&counts[b], 1ULL);
    }
}

extern "C" void histogram(const double *x, long n, long *counts, int nbins)
{
    const size_t bytes = sizeof(double) * (size_t)n, bins = sizeof(long) * (size_t)nbins;
    double *d_x;
    unsigned long long *d_counts;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, bins));
    CUDA_CHECK(cudaMemset(d_counts, 0, bins));
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    tally<<<BLOCKS, THREADS>>>(d_x, n, nbins, d_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(counts, d_counts, bins, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_counts));
}
