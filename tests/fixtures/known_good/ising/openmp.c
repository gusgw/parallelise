/* My own OpenMP routine for ising, written only to test the notebook; not a known-good routine.
 * Each random number is a hash of the seed, the sweep, the colour and the site, so the result does
 * not depend on the number of threads. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ising.h"

static uint64_t mix(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double uniform(uint64_t seed, uint64_t counter)
{
    uint64_t z = mix(seed + 0x9e3779b97f4a7c15ULL * (counter + 1));
    return (double)(mix(z) >> 11) * (1.0 / 9007199254740992.0);
}

void ising(long side, double beta, long equilibration, long sweeps, long seed, double *energy,
           double *magnetisation)
{
    signed char *spin = malloc((size_t)side * (size_t)side);
    if (spin == NULL) {
        *energy = 0.0;
        *magnetisation = 0.0;
        return;
    }
    memset(spin, 1, (size_t)side * (size_t)side);
    double accept4 = exp(-4.0 * beta);
    double accept8 = exp(-8.0 * beta);
    double energy_total = 0.0, magnetisation_total = 0.0;
    double sites = (double)side * (double)side;
    for (long sweep = 0; sweep < equilibration + sweeps; sweep++) {
        for (long colour = 0; colour < 2; colour++) {
            uint64_t stream = (uint64_t)seed * 0x2545f4914f6cdd1dULL + (uint64_t)(2 * sweep + colour);
#pragma omp parallel for schedule(static)
            for (long i = 0; i < side; i++) {
                long up = (i + side - 1) % side;
                long down = (i + 1) % side;
                for (long j = (i + colour) % 2; j < side; j += 2) {
                    long left = (j + side - 1) % side;
                    long right = (j + 1) % side;
                    int s = spin[i * side + j];
                    int around = spin[up * side + j] + spin[down * side + j]
                                 + spin[i * side + left] + spin[i * side + right];
                    int change = 2 * s * around;
                    if (change <= 0) {
                        spin[i * side + j] = (signed char)-s;
                    } else {
                        double chance = change == 4 ? accept4 : accept8;
                        if (uniform(stream, (uint64_t)(i * side + j)) < chance) {
                            spin[i * side + j] = (signed char)-s;
                        }
                    }
                }
            }
        }
        if (sweep >= equilibration) {
            long bonds = 0, total = 0;
#pragma omp parallel for reduction(+ : bonds, total) schedule(static)
            for (long i = 0; i < side; i++) {
                long up = (i + side - 1) % side;
                for (long j = 0; j < side; j++) {
                    long left = (j + side - 1) % side;
                    int s = spin[i * side + j];
                    bonds += s * (spin[up * side + j] + spin[i * side + left]);
                    total += s;
                }
            }
            energy_total -= (double)bonds / sites;
            magnetisation_total += fabs((double)total) / sites;
        }
    }
    *energy = energy_total / (double)sweeps;
    *magnetisation = magnetisation_total / (double)sweeps;
    free(spin);
}
