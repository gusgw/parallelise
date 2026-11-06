#include <cstdio>
#include <cstdlib>

#include "sparse_cg.h"

/* Known-good routine (validates the harness; never shown to an agent). The matrix and the
 * vectors stay on the graphics card. One thread for each row multiplies the matrix by the
 * search direction; the dot products are summed in blocks on the card and the block sums
 * added on the host, which decides when to stop. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCKS = 128, THREADS = 256;

__global__ static void multiply(const int *indptr, const int *indices, const double *data,
                                const double *p, double *q, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.0;
        for (int j = indptr[i]; j < indptr[i + 1]; j++)
            s += data[j] * p[indices[j]];
        q[i] = s;
    }
}

__global__ static void dot(const double *a, const double *b, long n, double *partial)
{
    __shared__ double sums[THREADS];
    double s = 0.0;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        s += a[i] * b[i];
    sums[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w)
            sums[threadIdx.x] += sums[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sums[0];
}

__global__ static void step_x_and_r(double *x, double *r, const double *p, const double *q,
                                    double alpha, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
    }
}

__global__ static void step_p(double *p, const double *r, double beta, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = r[i] + beta * p[i];
}

static double device_dot(const double *a, const double *b, long n, double *d_partial)
{
    dot<<<BLOCKS, THREADS>>>(a, b, n, d_partial);
    double partial[BLOCKS], s = 0.0;
    CUDA_CHECK(cudaMemcpy(partial, d_partial, sizeof(partial), cudaMemcpyDeviceToHost));
    for (int k = 0; k < BLOCKS; k++)
        s += partial[k];
    return s;
}

extern "C" void sparse_cg(const int *indptr, const int *indices, const double *data,
                          const double *b, double rtol, long max_iter, double *x, long np1,
                          long nnz, long n)
{
    const size_t vec = sizeof(double) * (size_t)n;
    const int row_blocks = (int)((n + THREADS - 1) / THREADS);
    int *d_indptr, *d_indices;
    double *d_data, *d_x, *d_r, *d_p, *d_q, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_indptr, sizeof(int) * (size_t)np1));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * (size_t)(nnz > 0 ? nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(double) * (size_t)(nnz > 0 ? nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_x, vec));
    CUDA_CHECK(cudaMalloc(&d_r, vec));
    CUDA_CHECK(cudaMalloc(&d_p, vec));
    CUDA_CHECK(cudaMalloc(&d_q, vec));
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(double) * BLOCKS));
    CUDA_CHECK(cudaMemcpy(d_indptr, indptr, sizeof(int) * (size_t)np1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, indices, sizeof(int) * (size_t)nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, data, sizeof(double) * (size_t)nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, vec));
    CUDA_CHECK(cudaMemcpy(d_r, b, vec, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, b, vec, cudaMemcpyHostToDevice));

    double rr = device_dot(d_r, d_r, n, d_partial);
    const double limit = rtol * rtol * rr;
    for (long step = 0; step < max_iter && rr > limit; step++) {
        multiply<<<row_blocks, THREADS>>>(d_indptr, d_indices, d_data, d_p, d_q, n);
        const double alpha = rr / device_dot(d_p, d_q, n, d_partial);
        step_x_and_r<<<row_blocks, THREADS>>>(d_x, d_r, d_p, d_q, alpha, n);
        const double rr_new = device_dot(d_r, d_r, n, d_partial);
        step_p<<<row_blocks, THREADS>>>(d_p, d_r, rr_new / rr, n);
        rr = rr_new;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(x, d_x, vec, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_partial));
}
