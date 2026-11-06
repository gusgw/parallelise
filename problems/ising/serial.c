/* The serial routine of ising.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [32, 64, 128, 512]. */
#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

/* Simple random number generator (LCG) to match the statistical behavior */
static uint64_t rng_state;

static void rng_seed(long seed) {
    rng_state = (uint64_t)seed;
    /* Warm up the RNG */
    for (int i = 0; i < 10; i++) {
        rng_state = rng_state * 6364136223846793005ULL + 1;
    }
}

static double rng_random(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1;
    /* Return value in [0, 1) */
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}

void ising(long side, double beta, long equilibration, long sweeps, long seed, double *energy, double *magnetisation) {
    long total_sweeps = equilibration + sweeps;
    long n_spins = side * side;
    
    /* Allocate spins array (1 for up, -1 for down) */
    int8_t *spins = malloc(n_spins * sizeof(int8_t));
    
    /* Initialize all spins to +1 */
    for (long i = 0; i < n_spins; i++) {
        spins[i] = 1;
    }
    
    /* Precompute colours (chessboard pattern): (row + col) % 2 */
    int8_t *colours = malloc(n_spins * sizeof(int8_t));
    for (long row = 0; row < side; row++) {
        for (long col = 0; col < side; col++) {
            colours[row * side + col] = (row + col) % 2;
        }
    }
    
    /* Initialize random number generator */
    rng_seed(seed);
    
    double energy_sum = 0.0;
    double magnetisation_sum = 0.0;
    
    for (long sweep = 0; sweep < total_sweeps; sweep++) {
        /* Two sublattice updates (colour 0 and colour 1) */
        for (int colour = 0; colour < 2; colour++) {
            for (long row = 0; row < side; row++) {
                for (long col = 0; col < side; col++) {
                    long idx = row * side + col;
                    
                    /* Only update spins of the current colour */
                    if (colours[idx] != colour) {
                        continue;
                    }
                    
                    /* Calculate neighbors with periodic boundary conditions */
                    long up = ((row - 1 + side) % side) * side + col;
                    long down = ((row + 1) % side) * side + col;
                    long left = row * side + ((col - 1 + side) % side);
                    long right = row * side + ((col + 1) % side);
                    
                    int8_t neighbours = spins[up] + spins[down] + spins[left] + spins[right];
                    
                    /* Energy change if spin were flipped: 2 * spin * neighbours */
                    double change = 2.0 * spins[idx] * neighbours;
                    
                    /* Metropolis criterion */
                    if (rng_random() < exp(-beta * change)) {
                        spins[idx] = -spins[idx];
                    }
                }
            }
        }
        
        /* Measure after equilibration */
        if (sweep >= equilibration) {
            /* Calculate bonds: spins * (right_neighbor + down_neighbor) */
            double bonds_sum = 0.0;
            for (long row = 0; row < side; row++) {
                for (long col = 0; col < side; col++) {
                    long idx = row * side + col;
                    long down = ((row + 1) % side) * side + col;
                    long right = row * side + ((col + 1) % side);
                    bonds_sum += spins[idx] * (spins[down] + spins[right]);
                }
            }
            energy_sum -= bonds_sum / n_spins;
            
            /* Calculate absolute magnetisation */
            long spin_sum = 0;
            for (long i = 0; i < n_spins; i++) {
                spin_sum += spins[i];
            }
            magnetisation_sum += fabs((double)spin_sum) / n_spins;
        }
    }
    
    *energy = energy_sum / sweeps;
    *magnetisation = magnetisation_sum / sweeps;
    
    free(spins);
    free(colours);
}
