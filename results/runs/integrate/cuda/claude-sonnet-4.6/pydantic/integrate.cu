/*
 * integrate.cu — GPU midpoint-rule integration of 4/(1+x^2) over [a,b].
 *
 * Strategy:
 *   • A grid-stride __global__ kernel: each thread accumulates its own
 *     partial sum over the subintervals it owns.
 *   • A shared-memory per-block reduction collapses the per-thread partials
 *     into one double per block.
 *   • The host function copies the per-block results back, sums them, and
 *     returns the finished integral.
 */

#include <stdio.h>
#include <stdlib.h>

#include "integrate.h"

/* ------------------------------------------------------------------ */
/* Tuning knobs                                                         */
/* ------------------------------------------------------------------ */
#define BLOCK_SIZE 256   /* threads per block — must be a power of 2  */
#define GRID_SIZE  2048  /* blocks launched; large enough to saturate  */
                         /* any current GPU while keeping launch cheap */

/* ------------------------------------------------------------------ */
/* Kernel                                                               */
/* ------------------------------------------------------------------ */
__global__ static void integrate_kernel(double a, double h, long n,
                                        double *partial)
{
    /* Shared memory for the per-block reduction (one slot per thread). */
    __shared__ double sdata[BLOCK_SIZE];

    const long tid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x  * blockDim.x;

    /* Each thread accumulates its own partial sum (grid-stride loop). */
    double local = 0.0;
    for (long i = tid; i < n; i += stride) {
        double x = a + ((double)i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    /* Tree reduction within the block. */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    /* Thread 0 of each block writes the block's partial sum. */
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

/* ------------------------------------------------------------------ */
/* Host function (extern "C" so plain-C callers see the right symbol)  */
/* ------------------------------------------------------------------ */
extern "C"
double integrate(double a, double b, long n)
{
    const double h = (b - a) / (double)n;

    /* Allocate device memory for per-block partial sums. */
    double *d_partial = NULL;
    cudaError_t err = cudaMalloc((void **)&d_partial,
                                 GRID_SIZE * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    /* Launch the kernel. */
    integrate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(a, h, n, d_partial);

    /* Check for launch errors. */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_partial);
        return 0.0;
    }

    /* Wait for the kernel to finish and check for runtime errors. */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial);
        return 0.0;
    }

    /* Copy per-block partial sums back to the host. */
    double h_partial[GRID_SIZE];
    err = cudaMemcpy(h_partial, d_partial, GRID_SIZE * sizeof(double),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    /* Final reduction on the host: sum the per-block partials. */
    double total = 0.0;
    for (int i = 0; i < GRID_SIZE; i++) {
        total += h_partial[i];
    }

    return total * h;
}
