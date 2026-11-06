#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ising.h"

/* Known-good routine (validates the harness; never shown to an agent). Each half sweep
 * updates all the spins of one colour of the chessboard at once, one thread for each spin,
 * with random numbers made by mixing the bits of the seed, the sweep, the colour and the
 * spin's place. The result agrees with the notebook's function within statistical error. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int THREADS = 256;

struct Accept { double p[17]; }; /* exp(-beta * change) for each change from -8 to 8 */

__device__ static double uniform(unsigned long long key)
{
    unsigned long long z = (key + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

__global__ static void half_sweep(signed char *s, long side, int colour, unsigned long long key,
                                  Accept accept)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= side * side)
        return;
    const long r = i / side, c = i % side;
    if ((r + c) % 2 != colour)
        return;
    const int near = s[((r + side - 1) % side) * side + c] + s[((r + 1) % side) * side + c]
                   + s[r * side + (c + side - 1) % side] + s[r * side + (c + 1) % side];
    if (uniform(key + (unsigned long long)i) < accept.p[2 * s[i] * near + 8])
        s[i] = (signed char)-s[i];
}

__global__ static void measure(const signed char *s, long side, int *sums)
{
    __shared__ int bonds[THREADS], spins[THREADS];
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int b = 0, m = 0;
    if (i < side * side) {
        const long r = i / side, c = i % side;
        b = s[i] * (s[((r + 1) % side) * side + c] + s[r * side + (c + 1) % side]);
        m = s[i];
    }
    bonds[threadIdx.x] = b;
    spins[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) {
            bonds[threadIdx.x] += bonds[threadIdx.x + w];
            spins[threadIdx.x] += spins[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sums[0], bonds[0]);
        atomicAdd(&sums[1], spins[0]);
    }
}

extern "C" void ising(long side, double beta, long equilibration, long sweeps, long seed,
                      double *energy, double *magnetisation)
{
    const long n = side * side;
    const int blocks = (int)((n + THREADS - 1) / THREADS);
    Accept accept;
    for (int c = -8; c <= 8; c++)
        accept.p[c + 8] = exp(-beta * c);
    signed char *d_s;
    int *d_sums;
    CUDA_CHECK(cudaMalloc(&d_s, (size_t)n));
    CUDA_CHECK(cudaMemset(d_s, 1, (size_t)n));
    CUDA_CHECK(cudaMalloc(&d_sums, 2 * sizeof(int)));
    double e = 0.0, m = 0.0;
    for (long sweep = 0; sweep < equilibration + sweeps; sweep++) {
        for (int colour = 0; colour < 2; colour++) {
            const unsigned long long key =
                (((unsigned long long)seed * 1000003ULL + (unsigned long long)sweep) * 2ULL
                 + (unsigned long long)colour) * (unsigned long long)n;
            half_sweep<<<blocks, THREADS>>>(d_s, side, colour, key, accept);
        }
        if (sweep >= equilibration) {
            int sums[2];
            CUDA_CHECK(cudaMemset(d_sums, 0, sizeof(sums)));
            measure<<<blocks, THREADS>>>(d_s, side, d_sums);
            CUDA_CHECK(cudaMemcpy(sums, d_sums, sizeof(sums), cudaMemcpyDeviceToHost));
            e -= (double)sums[0] / (double)n;
            m += fabs((double)sums[1]) / (double)n;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_sums));
    *energy = e / (double)sweeps;
    *magnetisation = m / (double)sweeps;
}
