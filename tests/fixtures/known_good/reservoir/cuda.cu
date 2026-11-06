#include <cstdio>
#include <cstdlib>

#include "reservoir.h"
#define RES_CAP (20000.0)

/* Known-good routine (validates the harness; never shown to an agent). Any run of steps is a
 * shift followed by a clamp, summarised by three numbers (see the MPI routine). The array is
 * cut into one chunk for each thread: each thread folds its chunk into one such map, the host
 * applies the maps in turn to find the level entering each chunk, and each thread then runs
 * the plain recurrence over its chunk. The numbers are whole, so the result is exact. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int BLOCKS = 256, THREADS = 256;

struct Clamp { double d, lo, hi; };

__host__ __device__ static double clampd(double x, double lo, double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

__global__ static void fold(const double *a, long n, long chunk, Clamp *maps)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    Clamp acc = {0.0, -1.0e18, 1.0e18};
    for (long i = lo; i < hi; i++) { /* apply acc, then the step (a[i], 0, RES_CAP) */
        acc.d += a[i];
        acc.lo = clampd(acc.lo + a[i], 0.0, RES_CAP);
        acc.hi = clampd(acc.hi + a[i], 0.0, RES_CAP);
    }
    maps[t] = acc;
}

__global__ static void run(const double *a, double *out, long n, long chunk, const double *entry)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double level = entry[t];
    for (long i = lo; i < hi; i++) {
        level = clampd(level + a[i], 0.0, RES_CAP);
        out[i] = level;
    }
}

extern "C" void reservoir(const double *a, double *out, long n)
{
    const long threads = (long)BLOCKS * THREADS, chunk = (n + threads - 1) / threads;
    const size_t bytes = sizeof(double) * (size_t)n;
    double *d_a, *d_out, *d_entry;
    Clamp *d_maps;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_maps, sizeof(Clamp) * (size_t)threads));
    CUDA_CHECK(cudaMalloc(&d_entry, sizeof(double) * (size_t)threads));
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    fold<<<BLOCKS, THREADS>>>(d_a, n, chunk, d_maps);
    CUDA_CHECK(cudaGetLastError());

    Clamp *maps = (Clamp *)malloc(sizeof(Clamp) * (size_t)threads);
    double *entry = (double *)malloc(sizeof(double) * (size_t)threads), level = 0.0;
    CUDA_CHECK(cudaMemcpy(maps, d_maps, sizeof(Clamp) * (size_t)threads, cudaMemcpyDeviceToHost));
    for (long t = 0; t < threads; t++) {
        entry[t] = level;
        level = clampd(level + maps[t].d, maps[t].lo, maps[t].hi);
    }
    CUDA_CHECK(cudaMemcpy(d_entry, entry, sizeof(double) * (size_t)threads,
                          cudaMemcpyHostToDevice));
    run<<<BLOCKS, THREADS>>>(d_a, d_out, n, chunk, d_entry);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    free(maps);
    free(entry);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_maps));
    CUDA_CHECK(cudaFree(d_entry));
}
