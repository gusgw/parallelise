/*
 * integrate.cu — GPU midpoint-rule integration of 4/(1+x^2) over [a,b].
 *
 * Design
 * ------
 *  • One __global__ grid-stride kernel: each thread accumulates its own
 *    partial sum over the subintervals it owns (stride = gridDim.x * blockDim.x).
 *  • Per-block shared-memory tree reduction collapses the block's partial sums
 *    into a single value that is written to a device-side partial-sums array.
 *  • The host function sums the per-block values on the CPU, multiplies by h,
 *    and returns the finished integral.
 *  • CUDA API and kernel errors are checked after every call.
 */

#include <stdio.h>
#include <stdlib.h>

#include "integrate.h"

/* ------------------------------------------------------------------ helpers */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ------------------------------------------------------------------ kernel  */

/*
 * Each thread computes a partial sum of the un-scaled terms
 *   f(x_i) = 4 / (1 + x_i^2),   x_i = a + (i + 0.5) * h
 * for every i in its grid-stride slice.  A shared-memory tree reduction then
 * produces one double per block, stored in d_partial[blockIdx.x].
 *
 * The final multiply by h is done on the host to avoid an extra pass.
 */
__global__ void integrate_kernel(double a, double h, long n, double *d_partial)
{
    /* Shared memory: one slot per thread in the block. */
    extern __shared__ double sdata[];

    long tid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x  * blockDim.x;

    /* Grid-stride accumulation — each thread sums its own slice. */
    double local_sum = 0.0;
    for (long i = tid; i < n; i += stride) {
        double x = a + ((double)i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    /* Tree reduction within the block. */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes the block result. */
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

/* ------------------------------------------------------------------ host API */

extern "C"
double integrate(double a, double b, long n)
{
    if (n <= 0) return 0.0;

    const double h = (b - a) / (double)n;

    /* Launch configuration: 256 threads/block, enough blocks to keep the GPU
     * busy but not so many that the host-side reduction dominates.            */
    const int THREADS = 256;
    /* Cap blocks so the partial-sum array stays small; 1024 blocks is plenty
     * for any n we care about (200 M / (256 * 1024) ≈ 763 iterations/thread). */
    int blocks = (int)((n + THREADS - 1) / THREADS);
    if (blocks > 1024) blocks = 1024;

    /* Allocate device partial-sums array. */
    double *d_partial = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_partial, (size_t)blocks * sizeof(double)));

    /* Launch kernel with dynamic shared memory = THREADS doubles per block. */
    size_t smem = (size_t)THREADS * sizeof(double);
    integrate_kernel<<<blocks, THREADS, smem>>>(a, h, n, d_partial);

    /* Check for launch errors. */
    CUDA_CHECK(cudaGetLastError());
    /* Wait for completion and check for runtime errors. */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy per-block partial sums back to host. */
    double *h_partial = (double *)malloc((size_t)blocks * sizeof(double));
    if (!h_partial) {
        fprintf(stderr, "integrate: host malloc failed\n");
        cudaFree(d_partial);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial,
                          (size_t)blocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_partial);

    /* Final reduction on the host: sum the per-block values. */
    double total = 0.0;
    for (int b_idx = 0; b_idx < blocks; b_idx++) {
        total += h_partial[b_idx];
    }
    free(h_partial);

    return total * h;
}
