#include <cstdio>
#include <cstdlib>

#include "integrate.h"

/* Reference solution (validates the harness -- NOT shown to the agents).
 * Grid-stride loop + shared-memory block reduction; per-block partials are summed on
 * the host (the block count is small). */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCKS = 256;
static const int THREADS = 256;

__global__ static void midpoint_kernel(double a, double h, long n, double *partial)
{
    __shared__ double sdata[THREADS];
    double s = 0.0;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double x = a + ((double)i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }
    sdata[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w)
            sdata[threadIdx.x] += sdata[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sdata[0];
}

extern "C" double integrate(double a, double b, long n)
{
    const double h = (b - a) / (double)n;
    double *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, BLOCKS * sizeof(double)));

    midpoint_kernel<<<BLOCKS, THREADS>>>(a, h, n, d_partial);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double partial[BLOCKS];
    CUDA_CHECK(cudaMemcpy(partial, d_partial, sizeof(partial), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    double sum = 0.0;
    for (int i = 0; i < BLOCKS; i++)
        sum += partial[i];
    return sum * h;
}
