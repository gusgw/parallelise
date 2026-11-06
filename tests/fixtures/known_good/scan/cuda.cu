#include <cstdio>
#include <cstdlib>

#include "scan.h"

/* Known-good routine (validates the harness; never shown to an agent). The array is cut into
 * one chunk for each thread. Each thread sums its chunk; the host turns the chunk sums into
 * the running total entering each chunk; and each thread then scans its chunk from there. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCKS = 256, THREADS = 256;

__global__ static void chunk_sums(const double *x, long n, long chunk, double *sums)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double s = 0.0;
    for (long i = lo; i < hi; i++)
        s += x[i];
    sums[t] = s;
}

__global__ static void chunk_scans(const double *x, double *out, long n, long chunk,
                                   const double *entry)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double s = entry[t];
    for (long i = lo; i < hi; i++) {
        s += x[i];
        out[i] = s;
    }
}

extern "C" void scan(const double *x, double *out, long n)
{
    const long threads = (long)BLOCKS * THREADS, chunk = (n + threads - 1) / threads;
    const size_t bytes = sizeof(double) * (size_t)n, small = sizeof(double) * (size_t)threads;
    double *d_x, *d_out, *d_sums;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_sums, small));
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    chunk_sums<<<BLOCKS, THREADS>>>(d_x, n, chunk, d_sums);
    CUDA_CHECK(cudaGetLastError());

    double *sums = (double *)malloc(small), total = 0.0;
    CUDA_CHECK(cudaMemcpy(sums, d_sums, small, cudaMemcpyDeviceToHost));
    for (long t = 0; t < threads; t++) {
        const double s = sums[t];
        sums[t] = total;
        total += s;
    }
    CUDA_CHECK(cudaMemcpy(d_sums, sums, small, cudaMemcpyHostToDevice));
    chunk_scans<<<BLOCKS, THREADS>>>(d_x, d_out, n, chunk, d_sums);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    free(sums);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sums));
}
