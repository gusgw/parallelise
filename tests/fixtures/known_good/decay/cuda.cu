#include <cstdio>
#include <cstdlib>

#include "decay.h"
#define DECAY_A (-2.0)
#define DECAY_B (-1.0)
#define DECAY_NEG (-1.0e9)

/* Known-good routine (validates the harness; never shown to an agent). One step of the
 * recurrence is an affine map in maximum-and-addition arithmetic, and two such maps compose
 * into one (see the MPI routine). The array is cut into one chunk for each thread: each
 * thread folds its chunk into one map, the host applies the maps in turn to find the pair of
 * scores entering each chunk, and each thread then runs the plain recurrence over its chunk.
 * The numbers are whole, so the result is exact. */

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

struct Map { double M[2][2]; double c[2]; };

__host__ __device__ static void apply_map(const Map &f, const double *s, double *out)
{
    for (int i = 0; i < 2; i++) {
        double r = f.M[i][0] + s[0], rb = f.M[i][1] + s[1];
        if (rb > r) r = rb;
        out[i] = f.c[i] > r ? f.c[i] : r;
    }
}

/* h = f o g (do g first, then f). */
__device__ static Map compose(const Map &f, const Map &g)
{
    Map h;
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 2; k++) {
            double a = f.M[i][0] + g.M[0][k], b = f.M[i][1] + g.M[1][k];
            h.M[i][k] = a > b ? a : b;
        }
        double a = f.M[i][0] + g.c[0], b = f.M[i][1] + g.c[1], m = a > b ? a : b;
        h.c[i] = m > f.c[i] ? m : f.c[i];
    }
    return h;
}

__global__ static void fold(const double *w, long n, long chunk, Map *maps)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    Map acc = {{{0.0, DECAY_NEG}, {DECAY_NEG, 0.0}}, {DECAY_NEG, DECAY_NEG}};
    for (long i = lo; i < hi; i++) {
        Map f = {{{DECAY_A, DECAY_B}, {0.0, DECAY_NEG}}, {w[i], DECAY_NEG}};
        acc = compose(f, acc);
    }
    maps[t] = acc;
}

__global__ static void run(const double *w, double *out, long n, long chunk, const double *entry)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long lo = t * chunk, hi = lo + chunk < n ? lo + chunk : n;
    double last = entry[2 * t], before_last = entry[2 * t + 1];
    for (long i = lo; i < hi; i++) {
        double x = DECAY_A + last, xb = DECAY_B + before_last;
        if (xb > x) x = xb;
        if (w[i] > x) x = w[i];
        out[i] = x;
        before_last = last;
        last = x;
    }
}

extern "C" void decay(const double *w, double *out, long n)
{
    const long threads = (long)BLOCKS * THREADS, chunk = (n + threads - 1) / threads;
    const size_t bytes = sizeof(double) * (size_t)n;
    double *d_w, *d_out, *d_entry;
    Map *d_maps;
    CUDA_CHECK(cudaMalloc(&d_w, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_maps, sizeof(Map) * (size_t)threads));
    CUDA_CHECK(cudaMalloc(&d_entry, 2 * sizeof(double) * (size_t)threads));
    CUDA_CHECK(cudaMemcpy(d_w, w, bytes, cudaMemcpyHostToDevice));
    fold<<<BLOCKS, THREADS>>>(d_w, n, chunk, d_maps);
    CUDA_CHECK(cudaGetLastError());

    Map *maps = (Map *)malloc(sizeof(Map) * (size_t)threads);
    double *entry = (double *)malloc(2 * sizeof(double) * (size_t)threads);
    CUDA_CHECK(cudaMemcpy(maps, d_maps, sizeof(Map) * (size_t)threads, cudaMemcpyDeviceToHost));
    double s[2] = {DECAY_NEG, DECAY_NEG};
    for (long t = 0; t < threads; t++) {
        entry[2 * t] = s[0];
        entry[2 * t + 1] = s[1];
        double next[2];
        apply_map(maps[t], s, next);
        s[0] = next[0];
        s[1] = next[1];
    }
    CUDA_CHECK(cudaMemcpy(d_entry, entry, 2 * sizeof(double) * (size_t)threads,
                          cudaMemcpyHostToDevice));
    run<<<BLOCKS, THREADS>>>(d_w, d_out, n, chunk, d_entry);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
    free(maps);
    free(entry);
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_maps));
    CUDA_CHECK(cudaFree(d_entry));
}
