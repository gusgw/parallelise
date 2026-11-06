#include <cstdio>
#include <cstdlib>

#include "compact.h"

/* Known-good routine (validates the harness; never shown to an agent). The array is cut into
 * one chunk for each thread. Each thread counts the elements of its chunk that exceed the
 * threshold; the host turns the counts into the place where each chunk's indices begin; and
 * each thread then writes its chunk's indices there, in order. */

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

__global__ static void count(const double *x, double threshold, long n, long chunk, long *counts)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    long c = 0;
    for (long i = lo; i < hi; i++)
        c += x[i] > threshold;
    counts[t] = c;
}

__global__ static void pick(const double *x, double threshold, long n, long chunk,
                            const long *starts, long *indices)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    long k = starts[t];
    for (long i = lo; i < hi; i++)
        if (x[i] > threshold)
            indices[k++] = i;
}

extern "C" void compact(const double *x, double threshold, long *indices, long *indices_count,
                        long n)
{
    const long threads = (long)BLOCKS * THREADS, chunk = (n + threads - 1) / threads;
    const size_t bytes = sizeof(double) * (size_t)n, small = sizeof(long) * (size_t)threads;
    double *d_x;
    long *d_counts, *d_indices;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, small));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(long) * (size_t)n));
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    count<<<BLOCKS, THREADS>>>(d_x, threshold, n, chunk, d_counts);
    CUDA_CHECK(cudaGetLastError());

    long *starts = (long *)malloc(small), total = 0;
    CUDA_CHECK(cudaMemcpy(starts, d_counts, small, cudaMemcpyDeviceToHost));
    for (long t = 0; t < threads; t++) {
        const long c = starts[t];
        starts[t] = total;
        total += c;
    }
    CUDA_CHECK(cudaMemcpy(d_counts, starts, small, cudaMemcpyHostToDevice));
    pick<<<BLOCKS, THREADS>>>(d_x, threshold, n, chunk, d_counts, d_indices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(indices, d_indices, sizeof(long) * (size_t)total,
                          cudaMemcpyDeviceToHost));
    *indices_count = total;
    free(starts);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_indices));
}
