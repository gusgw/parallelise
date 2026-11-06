#include <cstdio>
#include <cstdlib>

#include "jacobi.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Two device
 * buffers alternate; one 2D-grid kernel per sweep updates the interior; boundaries are
 * present in both buffers from the initial copies and are never written. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ static void sweep_kernel(const double *cur, double *next, long n)
{
    const long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= n - 1 || j < 1 || j >= n - 1)
        return;
    next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] + cur[(i + 1) * n + j]
                              + cur[i * n + j - 1] + cur[i * n + j + 1]);
}

extern "C" void jacobi(double *u, long n, int iters)
{
    const size_t bytes = sizeof(double) * (size_t)n * n;
    double *da = nullptr, *db = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMemcpy(da, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, u, bytes, cudaMemcpyHostToDevice)); /* boundaries */

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + 15) / 16), (unsigned)((n + 15) / 16));
    double *cur = da, *next = db;
    for (int it = 0; it < iters; it++) {
        sweep_kernel<<<grid, block>>>(cur, next, n);
        double *t = cur; cur = next; next = t;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(u, cur, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
}
