#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "md.h"

/* Known-good routine (validates the harness; never shown to an agent). Every process holds
 * every position and velocity and moves them all in the same way, but computes the forces on
 * its own block of particles only. The forces are shared after each step, and process 0 sums
 * the energy and the virial. */

static void forces_on(const double *x, double *f, long lo, long hi, long n, double box,
                      double cutoff, double *sums)
{
    double potential = 0.0, virial = 0.0;
    for (long i = lo; i < hi; i++) {
        double fi[3] = {0.0, 0.0, 0.0};
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
            double inv2 = 1.0 / r2, inv6 = inv2 * inv2 * inv2;
            double slope = 24.0 * inv2 * inv6 * (2.0 * inv6 - 1.0);
            potential += 2.0 * inv6 * (inv6 - 1.0);
            virial += 0.5 * slope * r2;
            for (int k = 0; k < 3; k++)
                fi[k] += slope * d[k];
        }
        for (int k = 0; k < 3; k++)
            f[3 * i + k] = fi[k];
    }
    sums[0] = potential;
    sums[1] = virial;
}

void md(const double *positions, const double *velocities, double box, double dt, long steps,
        double cutoff, double *potential, double *pressure, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *counts = malloc(sizeof(int) * (size_t)size), *starts = malloc(sizeof(int) * (size_t)size);
    for (int r = 0; r < size; r++) {
        starts[r] = (int)(3 * (n * r / size));
        counts[r] = (int)(3 * (n * (r + 1) / size)) - starts[r];
    }
    const long lo = n * rank / size, hi = n * (rank + 1) / size;
    const size_t bytes = sizeof(double) * (size_t)(3 * n);
    double *x = malloc(bytes), *v = malloc(bytes), *f = malloc(bytes);
    memcpy(x, positions, bytes);
    memcpy(v, velocities, bytes);

    double sums[2], total[2], potential_sum = 0.0, pressure_sum = 0.0;
    forces_on(x, f, lo, hi, n, box, cutoff, sums);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, f, counts, starts, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    for (long step = 0; step < steps; step++) {
        for (long k = 0; k < 3 * n; k++) {
            v[k] += 0.5 * dt * f[k];
            x[k] += dt * v[k];
        }
        forces_on(x, f, lo, hi, n, box, cutoff, sums);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, f, counts, starts, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        for (long k = 0; k < 3 * n; k++)
            v[k] += 0.5 * dt * f[k];
        if (step >= steps / 2) {
            MPI_Reduce(sums, total, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            double kinetic = 0.0;
            for (long k = 0; k < 3 * n; k++)
                kinetic += 0.5 * v[k] * v[k];
            potential_sum += total[0] / (double)n;
            pressure_sum += (2.0 * kinetic + total[1]) / (3.0 * box * box * box);
        }
    }
    if (rank == 0) {
        *potential = potential_sum / (double)(steps - steps / 2);
        *pressure = pressure_sum / (double)(steps - steps / 2);
    }
    free(x);
    free(v);
    free(f);
    free(counts);
    free(starts);
}
