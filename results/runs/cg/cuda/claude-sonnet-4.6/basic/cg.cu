/*
 * cg.cu  –  GPU conjugate gradient for the 2-D Poisson system.
 *
 * Implements the signature declared in cg.h (extern "C" void cgsolve(...)).
 *
 * Strategy
 * --------
 *  - kernel_apply_A   : grid-stride 2-D kernel, applies the 5-point stencil q = A p
 *  - kernel_axpy      : grid-stride 1-D kernel, y[k] += alpha * x[k]
 *  - kernel_axpy_neg  : grid-stride 1-D kernel, y[k] -= alpha * x[k]
 *  - kernel_update_p  : grid-stride 1-D kernel, p[k] = r[k] + beta * p[k]
 *  - kernel_dot       : grid-stride 1-D reduction kernel; each block writes one
 *                       partial sum into a device array; the host sums the partials.
 *
 * All dot products are accumulated in double precision.  The host drives the CG
 * loop, computing alpha and beta from the partial sums returned by kernel_dot.
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

#define BLOCK 256   /* threads per block for 1-D kernels          */
#define TILE  16    /* tile edge for the 2-D stencil kernel        */

/* ------------------------------------------------------------------ kernels */

/* 5-point Poisson stencil  q = A p  on an m x m grid (row-major). */
__global__ static void kernel_apply_A(const double * __restrict__ p,
                                      double       * __restrict__ q,
                                      long m)
{
    long total = m * m;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        long i = idx / m;
        long j = idx % m;
        double s = 4.0 * p[idx];
        if (i > 0)     s -= p[idx - m];
        if (i < m - 1) s -= p[idx + m];
        if (j > 0)     s -= p[idx - 1];
        if (j < m - 1) s -= p[idx + 1];
        q[idx] = s;
    }
}

/* x[k] += alpha * y[k] */
__global__ static void kernel_axpy(double * __restrict__ x,
                                   const double * __restrict__ y,
                                   double alpha, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)gridDim.x * blockDim.x)
        x[k] += alpha * y[k];
}

/* x[k] -= alpha * y[k] */
__global__ static void kernel_axpy_neg(double * __restrict__ x,
                                       const double * __restrict__ y,
                                       double alpha, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)gridDim.x * blockDim.x)
        x[k] -= alpha * y[k];
}

/* p[k] = r[k] + beta * p[k] */
__global__ static void kernel_update_p(double * __restrict__ p,
                                       const double * __restrict__ r,
                                       double beta, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)gridDim.x * blockDim.x)
        p[k] = r[k] + beta * p[k];
}

/*
 * Dot product: each block reduces its share into partials[blockIdx.x].
 * The host sums the (small) partials array.
 */
__global__ static void kernel_dot(const double * __restrict__ u,
                                  const double * __restrict__ v,
                                  double       * __restrict__ partials,
                                  long n)
{
    __shared__ double sdata[BLOCK];

    long tid  = threadIdx.x;
    double acc = 0.0;

    for (long k = (long)blockIdx.x * blockDim.x + tid;
         k < n;
         k += (long)gridDim.x * blockDim.x)
        acc += u[k] * v[k];

    sdata[tid] = acc;
    __syncthreads();

    /* tree reduction within the block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

/* ------------------------------------------------------------------ host dot */

/*
 * Launch kernel_dot and return the scalar result.
 * d_partials must be pre-allocated on the device with at least nblocks doubles.
 * h_partials is a host-side scratch buffer of the same size.
 */
static double gpu_dot(const double *d_u, const double *d_v,
                      double *d_partials, double *h_partials,
                      int nblocks, long n)
{
    kernel_dot<<<nblocks, BLOCK>>>(d_u, d_v, d_partials, n);
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

    /* ---- device buffers ---- */
    double *d_x, *d_b, *d_r, *d_p, *d_q;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_r, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_p, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(double) * n));

    /* ---- copy b to device; initialise x = 0, r = b, p = b ---- */
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, sizeof(double) * n));
    CUDA_CHECK(cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    /* ---- partial-sum scratch ---- */
    int nblocks = (int)((n + BLOCK - 1) / BLOCK);
    /* cap so the partials array stays small */
    if (nblocks > 1024) nblocks = 1024;

    double *d_partials;
    CUDA_CHECK(cudaMalloc(&d_partials, sizeof(double) * nblocks));
    double *h_partials = (double *)malloc(sizeof(double) * nblocks);
    if (!h_partials) { fprintf(stderr, "malloc failed\n"); exit(1); }

    /* ---- rr = dot(r, r) ---- */
    double rr = gpu_dot(d_r, d_r, d_partials, h_partials, nblocks, n);

    /* ---- CG iterations ---- */
    int grid1d = nblocks;   /* reuse same block count for 1-D kernels */

    for (int it = 0; it < iters; it++) {
        /* q = A p */
        kernel_apply_A<<<grid1d, BLOCK>>>(d_p, d_q, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* alpha = rr / dot(p, q) */
        double pq   = gpu_dot(d_p, d_q, d_partials, h_partials, nblocks, n);
        double alpha = rr / pq;

        /* x += alpha * p */
        kernel_axpy<<<grid1d, BLOCK>>>(d_x, d_p, alpha, n);
        CUDA_CHECK(cudaGetLastError());

        /* r -= alpha * q */
        kernel_axpy_neg<<<grid1d, BLOCK>>>(d_r, d_q, alpha, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* rr_new = dot(r, r) */
        double rr_new = gpu_dot(d_r, d_r, d_partials, h_partials, nblocks, n);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        kernel_update_p<<<grid1d, BLOCK>>>(d_p, d_r, beta, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---- copy result back ---- */
    CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost));

    /* ---- free device buffers ---- */
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_partials));
    free(h_partials);
}
