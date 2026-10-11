#include <cstdio>
#include <cstdlib>

#include "cg.h"

/* Reference solution (validates the harness -- NOT shown to the agents). The vectors
 * x, r, p and q live on the device for the whole solve; the host drives the iterations
 * and holds only the scalars. A 2D-grid kernel applies the 5-point operator, grid-stride
 * kernels do the vector updates, and the dot products reduce per block in shared memory;
 * the block partials come back to the host and are summed there (simple and correct). */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define BLOCK 256     /* threads per block for the 1D kernels (power of two) */
#define NBLOCKS 128   /* fixed block count for the grid-stride dot kernels */

__global__ static void apply_a_kernel(const double *p, double *q, long m)
{
    const long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= m)
        return;
    const long idx = i * m + j;
    double s = 4.0 * p[idx];
    if (i > 0)     s -= p[idx - m];
    if (i < m - 1) s -= p[idx + m];
    if (j > 0)     s -= p[idx - 1];
    if (j < m - 1) s -= p[idx + 1];
    q[idx] = s;
}

/* x += alpha p and r -= alpha q in one pass over the vectors. */
__global__ static void update_xr_kernel(double *x, double *r, const double *p,
                                        const double *q, double alpha, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long)gridDim.x * blockDim.x) {
        x[k] += alpha * p[k];
        r[k] -= alpha * q[k];
    }
}

__global__ static void update_p_kernel(double *p, const double *r, double beta, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long)gridDim.x * blockDim.x)
        p[k] = r[k] + beta * p[k];
}

/* Each block reduces its grid-stride slice of a.b in shared memory and writes one
 * partial sum; the host sums the NBLOCKS partials. */
__global__ static void dot_kernel(const double *a, const double *b, double *partials,
                                  long n)
{
    __shared__ double sh[BLOCK];
    double s = 0.0;
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (long)gridDim.x * blockDim.x)
        s += a[k] * b[k];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w)
            sh[threadIdx.x] += sh[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sh[0];
}

/* Launch the dot kernel and finish the reduction on the host. The device-to-host copy
 * of the partials also synchronises with the kernels launched so far. */
static double device_dot(const double *da, const double *db, double *dpartials,
                         double *hpartials, long n)
{
    dot_kernel<<<NBLOCKS, BLOCK>>>(da, db, dpartials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hpartials, dpartials, sizeof(double) * NBLOCKS,
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < NBLOCKS; i++)
        s += hpartials[i];
    return s;
}

extern "C" void cgsolve(double *x, const double *b, long m, int iters)
{
    const long n = m * m;
    const size_t bytes = sizeof(double) * (size_t)n;
    double *dx = nullptr, *dr = nullptr, *dp = nullptr, *dq = nullptr;
    double *dpartials = nullptr;
    double hpartials[NBLOCKS];
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dr, bytes));
    CUDA_CHECK(cudaMalloc(&dp, bytes));
    CUDA_CHECK(cudaMalloc(&dq, bytes));
    CUDA_CHECK(cudaMalloc(&dpartials, sizeof(double) * NBLOCKS));

    /* x = 0 (all-zero bytes is the double 0.0); r = b; p = b */
    CUDA_CHECK(cudaMemset(dx, 0, bytes));
    CUDA_CHECK(cudaMemcpy(dr, b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp, dr, bytes, cudaMemcpyDeviceToDevice));

    double rr = device_dot(dr, dr, dpartials, hpartials, n);

    dim3 block2d(16, 16);
    dim3 grid2d((unsigned)((m + 15) / 16), (unsigned)((m + 15) / 16));
    for (int it = 0; it < iters; it++) {
        apply_a_kernel<<<grid2d, block2d>>>(dp, dq, m);
        CUDA_CHECK(cudaGetLastError());
        const double pq = device_dot(dp, dq, dpartials, hpartials, n);

        const double alpha = rr / pq;
        update_xr_kernel<<<NBLOCKS, BLOCK>>>(dx, dr, dp, dq, alpha, n);
        CUDA_CHECK(cudaGetLastError());
        const double rr_new = device_dot(dr, dr, dpartials, hpartials, n);

        const double beta = rr_new / rr;
        rr = rr_new;
        update_p_kernel<<<NBLOCKS, BLOCK>>>(dp, dr, beta, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(x, dx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dr));
    CUDA_CHECK(cudaFree(dp));
    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(dpartials));
}
