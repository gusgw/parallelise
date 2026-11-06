/* My own OpenMP routine for md, written only to test the notebook; not a known-good routine. */
#include <math.h>
#include <stdlib.h>

#include "md.h"

static void interactions(const double *positions, double *forces, long n, long d, double box,
                         double cutoff, double *potential, double *virial)
{
    double cutoff2 = cutoff * cutoff;
    double pot = 0.0, vir = 0.0;
#pragma omp parallel for reduction(+ : pot, vir) schedule(static)
    for (long i = 0; i < n; i++) {
        double f[3] = {0.0, 0.0, 0.0};
        for (long j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            double delta[3];
            double r2 = 0.0;
            for (long k = 0; k < d; k++) {
                double dx = positions[i * d + k] - positions[j * d + k];
                dx -= box * round(dx / box);
                delta[k] = dx;
                r2 += dx * dx;
            }
            if (r2 < cutoff2) {
                double inv_r2 = 1.0 / r2;
                double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                pot += 2.0 * inv_r6 * (inv_r6 - 1.0);
                double slope = 24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0);
                for (long k = 0; k < d; k++) {
                    f[k] += slope * delta[k];
                }
                vir += 0.5 * slope * r2;
            }
        }
        for (long k = 0; k < d; k++) {
            forces[i * d + k] = f[k];
        }
    }
    *potential = pot;
    *virial = vir;
}

void md(const double *positions, const double *velocities, double box, double dt, long steps,
        double cutoff, double *potential, double *pressure, long n)
{
    const long d = 3;
    double *x = malloc(sizeof(double) * (size_t)(n * d));
    double *v = malloc(sizeof(double) * (size_t)(n * d));
    double *f = malloc(sizeof(double) * (size_t)(n * d));
    if (x == NULL || v == NULL || f == NULL || d > 3) {
        free(x);
        free(v);
        free(f);
        *potential = 0.0;
        *pressure = 0.0;
        return;
    }
    for (long k = 0; k < n * d; k++) {
        x[k] = positions[k];
        v[k] = velocities[k];
    }
    double pot = 0.0, vir = 0.0;
    interactions(x, f, n, d, box, cutoff, &pot, &vir);
    double potential_sum = 0.0, pressure_sum = 0.0;
    double volume = pow(box, (double)d);
    for (long step = 0; step < steps; step++) {
#pragma omp parallel for
        for (long k = 0; k < n * d; k++) {
            v[k] += 0.5 * dt * f[k];
            x[k] += dt * v[k];
        }
        interactions(x, f, n, d, box, cutoff, &pot, &vir);
        double kinetic = 0.0;
#pragma omp parallel for reduction(+ : kinetic)
        for (long k = 0; k < n * d; k++) {
            v[k] += 0.5 * dt * f[k];
            kinetic += 0.5 * v[k] * v[k];
        }
        if (step >= steps / 2) {
            potential_sum += pot / (double)n;
            pressure_sum += (2.0 * kinetic + vir) / ((double)d * volume);
        }
    }
    double count = (double)(steps - steps / 2);
    *potential = potential_sum / count;
    *pressure = pressure_sum / count;
    free(x);
    free(v);
    free(f);
}
