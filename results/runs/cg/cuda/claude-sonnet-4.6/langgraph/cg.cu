/*
 * cg.cu  –  GPU conjugate-gradient solver for the 2-D Poisson system.
 *
 * Algorithm (from cg.ipynb / cg.h):
 *   x = 0;  r = b;  p = r;  rr = dot(r,r);
 *   for iters:
 *     q = A p                          (5-point stencil)
 *     alpha = rr / dot(p,q)
 *     x += alpha * p
 *     r -= alpha * q
 *     rr_new = dot(r,r)
 *     beta  = rr_new / rr
 *     rr    = rr_new
 *     p = r + beta * p
 *
 * Parallelism strategy
 * --------------------
 *   • kernel_apply_A   – 2-D grid, one thread per grid cell, applies the stencil.
 *   • kernel_axpy      – 1-D grid-stride, y += a*x  (or  y = x + a*y  for p-update).
 *   • kernel_dot       – 1-D grid-stride with per-block shared-memory reduction;
 *                        block partial sums are copied back to the host and summed
 *                        there (simple, correct, avoids atomics on FP64).
 *   • kernel_init      – initialise x=0, r=b, p=b in one pass.
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

/* Initialise device vectors: x=0, r=b, p=b */
__global__ static void kernel_init(double *x, double *r, double *p,
                                   const double *b, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)blockDim.x * gridDim.x)
    {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
    }
}

/* q = A p  (5-point Poisson stencil, Dirichlet BCs outside grid) */
__global__ static void kernel_apply_A(double *q, const double *p, long m)
{
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= m) return;

    long idx = i * m + j;
    double s = 4.0 * p[idx];
    if (i > 0)     s -= p[idx - m];
    if (i < m - 1) s -= p[idx + m];
    if (j > 0)     s -= p[idx - 1];
    if (j < m - 1) s -= p[idx + 1];
    q[idx] = s;
}

/* dot product: each block produces one partial sum stored in partials[blockIdx.x] */
__global__ static void kernel_dot(const double *u, const double *v,
                                  double *partials, long n)
{
    __shared__ double shmem[BLOCK];

    long tid  = threadIdx.x;
    double acc = 0.0;
    for (long k = (long)blockIdx.x * blockDim.x + tid;
         k < n;
         k += (long)blockDim.x * gridDim.x)
    {
        acc += u[k] * v[k];
    }
    shmem[tid] = acc;
    __syncthreads();

    /* tree reduction within the block */
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < (unsigned)s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shmem[0];
}

/* x += a * y  (in-place axpy on x) */
__global__ static void kernel_axpy(double *x, double a, const double *y, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)blockDim.x * gridDim.x)
    {
        x[k] += a * y[k];
    }
}

/* x -= a * y  (in-place, sign-flipped axpy) */
__global__ static void kernel_axpy_neg(double *x, double a, const double *y, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)blockDim.x * gridDim.x)
    {
        x[k] -= a * y[k];
    }
}

/* p = r + beta * p  (combined update to avoid a second pass) */
__global__ static void kernel_update_p(double *p, const double *r,
                                       double beta, long n)
{
    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < n;
         k += (long)blockDim.x * gridDim.x)
    {
        p[k] = r[k] + beta * p[k];
    }
}

/* ------------------------------------------------------------------ helpers */

/* Choose a grid size for 1-D kernels: cap at 65535 blocks. */
static int grid1d(long n)
{
    long g = (n + BLOCK - 1) / BLOCK;
    return (int)(g > 65535 ? 65535 : g);
}

/* Launch kernel_dot and return the scalar result (host-side reduction). */
static double device_dot(const double *d_u, const double *d_v,
                         double *d_partials, double *h_partials,
                         int nblocks, long n)
{
    kernel_dot<<<nblocks, BLOCK>>>(d_u, d_v, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_partials, d_partials,
                          sizeof(double) * nblocks, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int b = 0; b < nblocks; b++) s += h_partials[b];
    return s;
}

/* ------------------------------------------------------------------ public API */

extern "C"
void cgsolve(double *x, const double *b, long m, int iters)
{
    const long n = m * m;

    /* ---- device allocations ---- */
    double *d_x, *d_r, *d_p, *d_q, *d_b, *d_partials;
    CUDA_CHECK(cudaMalloc(&d_x,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_r,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_p,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_q,        sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_b,        sizeof(double) * n));

    /* partial-sum buffer: one double per block */
    int nblocks = grid1d(n);
    CUDA_CHECK(cudaMalloc(&d_partials, sizeof(double) * nblocks));
    double *h_partials = (double *)malloc(sizeof(double) * nblocks);
    if (!h_partials) { fprintf(stderr, "malloc failed\n"); exit(1); }

    /* ---- copy b to device ---- */
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(double) * n, cudaMemcpyHostToDevice));

    /* ---- initialise x=0, r=b, p=b ---- */
    kernel_init<<<nblocks, BLOCK>>>(d_x, d_r, d_p, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- rr = dot(r, r) ---- */
    double rr = device_dot(d_r, d_r, d_partials, h_partials, nblocks, n);

    /* ---- 2-D grid dimensions for the stencil kernel ---- */
    dim3 block2d(TILE, TILE);
    dim3 grid2d((unsigned int)((m + TILE - 1) / TILE),
                (unsigned int)((m + TILE - 1) / TILE));

    /* ---- CG iterations ---- */
    for (int it = 0; it < iters; it++) {

        /* q = A p */
        kernel_apply_A<<<grid2d, block2d>>>(d_q, d_p, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* alpha = rr / dot(p, q) */
        double pq   = device_dot(d_p, d_q, d_partials, h_partials, nblocks, n);
        double alpha = rr / pq;

        /* x += alpha * p */
        kernel_axpy<<<nblocks, BLOCK>>>(d_x, alpha, d_p, n);
        CUDA_CHECK(cudaGetLastError());

        /* r -= alpha * q */
        kernel_axpy_neg<<<nblocks, BLOCK>>>(d_r, alpha, d_q, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* rr_new = dot(r, r) */
        double rr_new = device_dot(d_r, d_r, d_partials, h_partials, nblocks, n);
        double beta   = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        kernel_update_p<<<nblocks, BLOCK>>>(d_p, d_r, beta, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ---- copy result back to host ---- */
    CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost));

    /* ---- free device memory ---- */
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_partials));
    free(h_partials);
}
