#include <cstdio>
#include <cstdlib>

#include "nbody.h"

/* Reference solution (validates the harness -- NOT shown to the agents). One thread per
 * body i in a grid-stride loop (no shared-memory tiling; the direct sum is enough here);
 * the host wrapper owns the device buffers and the copies. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ static void nbody_kernel(const double *px, const double *py,
                                    double *ax, double *ay, long n)
{
    const double eps2 = 0.0025;   /* softening 0.05 squared */
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double xi = px[i];
        const double yi = py[i];
        double sx = 0.0, sy = 0.0;
        for (long j = 0; j < n; j++) {
            if (j == i)
                continue;
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double r2 = dx * dx + dy * dy;
            const double d3 = (r2 + eps2) * sqrt(r2 + eps2);
            sx += dx / d3;
            sy += dy / d3;
        }
        ax[i] = sx;
        ay[i] = sy;
    }
}

extern "C" void nbody_accel(const double *px, const double *py,
                            double *ax, double *ay, long n)
{
    const size_t bytes = sizeof(double) * (size_t)n;
    double *dpx = nullptr, *dpy = nullptr, *dax = nullptr, *day = nullptr;
    CUDA_CHECK(cudaMalloc(&dpx, bytes));
    CUDA_CHECK(cudaMalloc(&dpy, bytes));
    CUDA_CHECK(cudaMalloc(&dax, bytes));
    CUDA_CHECK(cudaMalloc(&day, bytes));
    CUDA_CHECK(cudaMemcpy(dpx, px, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpy, py, bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    long grid = (n + block - 1) / block;
    if (grid < 1) grid = 1;
    if (grid > 4096) grid = 4096;   /* the grid-stride loop covers the rest */
    nbody_kernel<<<(unsigned)grid, block>>>(dpx, dpy, dax, day, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ax, dax, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay, day, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dpx));
    CUDA_CHECK(cudaFree(dpy));
    CUDA_CHECK(cudaFree(dax));
    CUDA_CHECK(cudaFree(day));
}
