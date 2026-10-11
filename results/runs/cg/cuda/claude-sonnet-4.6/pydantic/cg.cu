/*
 * cg.cu  --  Conjugate Gradient for the 2-D Poisson system on the GPU.
 *
 * Implements the exact signature declared in cg.h.
 * The algorithm mirrors the Python notebook exactly:
 *   x = 0;  r = b;  p = r;  rr = dot(r,r);
 *   for iters:
 *     q = A p
 *     alpha = rr / dot(p,q)
 *     x += alpha * p
 *     r -= alpha * q
 *     rr_new = dot(r,r)
 *     beta = rr_new / rr
 *     rr = rr_new
 *     p = r + beta * p
 *
 * Parallelism strategy
 * --------------------
 *   apply_A_kernel   : 2-D grid-stride kernel, each thread handles one (i,j) cell.
 *   axpy_kernel      : 1-D grid-stride kernel for vector updates (x+=a*p, r-=a*q, p=r+b*p).
 *   dot_kernel       : 1-D grid-stride kernel with per-block shared-memory reduction;
 *                      block partial sums are copied back to the host and summed there.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cg.h"

/* ------------------------------------------------------------------ helpers */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ------------------------------------------------------------------ kernels */

/* 5-point Poisson operator: q = A p */
__global__ static void apply_A_kernel(const double * __restrict__ p,
                                      double       * __restrict__ q,
                                      long m)
{
    long total = m * m;
    for (long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += (long)gridDim.x * blockDim.x)
    {
        long i = tid / m;
        long j = tid % m;
        double s = 4.0 * p[tid];
        if (i > 0)     s -= p[tid - m];
        if (i < m - 1) s -= p[tid + m];
        if (j > 0)     s -= p[tid - 1];
        if (j < m - 1) s -= p[tid + 1];
        q[tid] = s;
    }
}

/* x[k] += a * y[k] */
__global__ static void axpy_kernel(double *x, double a, const double *y, long n)
{
    for (long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += (long)gridDim.x * blockDim.x)
    {
        x[tid] += a * y[tid];
    }
}

/* x[k] -= a * y[k] */
__global__ static void axmy_kernel(double *x, double a, const double *y, long n)
{
    for (long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += (long)gridDim.x * blockDim.x)
    {
        x[tid] -= a * y[tid];
    }
}

/* p[k] = r[k] + beta * p[k] */
__global__ static void update_p_kernel(double *p, const double *r, double beta, long n)
{
    for (long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += (long)gridDim.x * blockDim.x)
    {
        p[tid] = r[tid] + beta * p[tid];
    }
}

/*
 * dot_kernel: each block reduces its share into one partial sum stored in
 * d_partials[blockIdx.x].  The host sums the partials.
 */
#define BLOCK_SIZE 256

__global__ static void dot_kernel(const double * __restrict__ u,
                                  const double * __restrict__ v,
                                  double       * __restrict__ partials,
                                  long n)
{
    __shared__ double sdata[BLOCK_SIZE];

    long tid  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step = (long)gridDim.x  * blockDim.x;
    int  lid  = threadIdx.x;

    double acc = 0.0;
    for (long k = tid; k < n; k += step)
        acc += u[k] * v[k];
    sdata[lid] = acc;
    __syncthreads();

    /* tree reduction within the block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        __syncthreads();
    }
    if (lid == 0) partials[blockIdx.x] = sdata[0];
}

/* ------------------------------------------------------------------ host dot */

static double gpu_dot(const double *d_u, const double *d_v, long n,
                      double *d_partials, double *h_partials, int nblocks)
{
    dot_kernel<<<nblocks, BLOCK_SIZE>>>(d_u, d_v, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partials, d_partials,
                          sizeof(double) * nblocks, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < nblocks; i++) s += h_partials[i];
    return s;
}

/* ------------------------------------------------------------------ cgsolve */

extern "C"
void cgsolve(double *x, const double *b, long m, int iters)
{
    long n = m * m;

    /* choose grid dimensions */
    const int THREADS = BLOCK_SIZE;
    int nblocks = (int)((n + THREADS - 1) / THREADS);
    if (nblocks > 1024) nblocks = 1024;   /* cap so partial array stays small */

    /* allocate device buffers */
    double *d_x, *d_b, *d_r, *d_p, *d_q, *d_partials;
    CUDA_CHECK(cudaMalloc(&d_x,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_b,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_r,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_p,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_q,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_partials, sizeof(double) * nblocks));

    /* host partial-sum buffer */
    double *h_partials = (double *)malloc(sizeof(double) * nblocks);
    if (!h_partials) { fprintf(stderr, "malloc failed\n"); exit(1); }

    /* copy b to device; initialise x = 0, r = b, p = b */
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, sizeof(double) * n));
    CUDA_CHECK(cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    double rr = gpu_dot(d_r, d_r, n, d_partials, h_partials, nblocks);

    for (int it = 0; it < iters; it++) {
        /* q = A p */
        apply_A_kernel<<<nblocks, THREADS>>>(d_p, d_q, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* alpha = rr / dot(p, q) */
        double pq   = gpu_dot(d_p, d_q, n, d_partials, h_partials, nblocks);
        double alpha = rr / pq;

        /* x += alpha * p */
        axpy_kernel<<<nblocks, THREADS>>>(d_x,  alpha, d_p, n);
        CUDA_CHECK(cudaGetLastError());

        /* r -= alpha * q */
        axmy_kernel<<<nblocks, THREADS>>>(d_r,  alpha, d_q, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* rr_new = dot(r, r) */
        double rr_new = gpu_dot(d_r, d_r, n, d_partials, h_partials, nblocks);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        update_p_kernel<<<nblocks, THREADS>>>(d_p, d_r, beta, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* copy result back to host */
    CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost));

    /* free device and host buffers */
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_partials));
    free(h_partials);
}
