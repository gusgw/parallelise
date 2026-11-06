#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "md.h"

/* Known-good routine (validates the harness; never shown to an agent). The particles stay on
 * the graphics card for the whole run. In each step one thread for each particle finds the
 * force on it and its shares of the potential energy and the virial; in the steps that are
 * measured, the host adds up the shares and the kinetic energy. */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int THREADS = 128;

__global__ static void kick_drift(double *x, double *v, const double *f, double dt, long count)
{
    const long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) {
        v[k] += 0.5 * dt * f[k];
        x[k] += dt * v[k];
    }
}

__global__ static void kick(double *v, const double *f, double dt, long count)
{
    const long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count)
        v[k] += 0.5 * dt * f[k];
}

__global__ static void forces(const double *x, double *f, double *shares, long n, double box,
                              double cutoff)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double fi[3] = {0.0, 0.0, 0.0}, potential = 0.0, virial = 0.0;
    for (long j = 0; j < n; j++) {
        if (j == i)
            continue;
        double d[3], r2 = 0.0;
        for (int k = 0; k < 3; k++) {
            d[k] = x[3 * i + k] - x[3 * j + k];
            d[k] -= box * nearbyint(d[k] / box);
            r2 += d[k] * d[k];
        }
        if (r2 >= cutoff * cutoff)
            continue;
        const double inv2 = 1.0 / r2, inv6 = inv2 * inv2 * inv2;
        const double slope = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0);
        potential += 2.0 * inv6 * (inv6 - 1.0);
        virial += 0.5 * slope * r2;
        for (int k = 0; k < 3; k++)
            fi[k] += slope * d[k];
    }
    for (int k = 0; k < 3; k++)
        f[3 * i + k] = fi[k];
    shares[2 * i] = potential;
    shares[2 * i + 1] = virial;
}

extern "C" void md(const double *positions, const double *velocities, double box, double dt,
                   long steps, double cutoff, double *potential, double *pressure, long n)
{
    const long count = 3 * n;
    const size_t bytes = sizeof(double) * (size_t)count, share_bytes = 2 * sizeof(double) * (size_t)n;
    const int particle_blocks = (int)((n + THREADS - 1) / THREADS);
    const int value_blocks = (int)((count + THREADS - 1) / THREADS);
    double *d_x, *d_v, *d_f, *d_shares;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_f, bytes));
    CUDA_CHECK(cudaMalloc(&d_shares, share_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, positions, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, velocities, bytes, cudaMemcpyHostToDevice));
    forces<<<particle_blocks, THREADS>>>(d_x, d_f, d_shares, n, box, cutoff);

    double *v = (double *)malloc(bytes), *shares = (double *)malloc(share_bytes);
    double potential_sum = 0.0, pressure_sum = 0.0;
    for (long step = 0; step < steps; step++) {
        kick_drift<<<value_blocks, THREADS>>>(d_x, d_v, d_f, dt, count);
        forces<<<particle_blocks, THREADS>>>(d_x, d_f, d_shares, n, box, cutoff);
        kick<<<value_blocks, THREADS>>>(d_v, d_f, dt, count);
        if (step >= steps / 2) {
            CUDA_CHECK(cudaMemcpy(v, d_v, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(shares, d_shares, share_bytes, cudaMemcpyDeviceToHost));
            double kinetic = 0.0, pot = 0.0, vir = 0.0;
            for (long k = 0; k < count; k++)
                kinetic += 0.5 * v[k] * v[k];
            for (long i = 0; i < n; i++) {
                pot += shares[2 * i];
                vir += shares[2 * i + 1];
            }
            potential_sum += pot / (double)n;
            pressure_sum += (2.0 * kinetic + vir) / (3.0 * box * box * box);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    *potential = potential_sum / (double)(steps - steps / 2);
    *pressure = pressure_sum / (double)(steps - steps / 2);
    free(v);
    free(shares);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_shares));
}
