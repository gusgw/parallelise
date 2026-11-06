/* The serial routine of md.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [125, 216, 343]. */
#include "md.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void interactions(const double *positions, double box, double cutoff, long n, long d,
                         double *forces, double *potential_out, double *virial_out) {
    double potential = 0.0;
    double virial = 0.0;
    double cutoff_sq = cutoff * cutoff;
    
    // Initialize forces to zero
    for (long i = 0; i < n * d; i++) {
        forces[i] = 0.0;
    }
    
    // Process in blocks of 256
    for (long start = 0; start < n; start += 256) {
        long end = start + 256;
        if (end > n) end = n;
        
        for (long i = start; i < end; i++) {
            for (long j = 0; j < n; j++) {
                // Skip self-interaction
                if (i == j) continue;
                
                // Calculate delta with periodic boundary conditions
                double delta[3];  // d is at most 3
                double r2 = 0.0;
                
                for (long k = 0; k < d; k++) {
                    delta[k] = positions[i * d + k] - positions[j * d + k];
                    // Apply periodic boundary: delta -= box * round(delta / box)
                    delta[k] -= box * round(delta[k] / box);
                    r2 += delta[k] * delta[k];
                }
                
                if (r2 < cutoff_sq && r2 > 0.0) {
                    double inv_r2 = 1.0 / r2;
                    double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    
                    // Potential: 4 * (r^-12 - r^-6) = 4 * inv_r6 * (inv_r6 - 1)
                    // Each pair appears twice in the sum, so we add half each time
                    potential += 2.0 * inv_r6 * (inv_r6 - 1.0);
                    
                    // slope = 24 * inv_r2 * inv_r6 * (2 * inv_r6 - 1)
                    double slope = 24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0);
                    
                    // Add to virial: 0.5 * slope * r2 (but we sum full contribution here)
                    virial += 0.5 * slope * r2;
                    
                    // Update force on particle i only (particle j will be updated when its block is processed)
                    for (long k = 0; k < d; k++) {
                        forces[i * d + k] += slope * delta[k];
                    }
                }
            }
        }
    }
    
    *potential_out = potential;
    *virial_out = virial;
}

void md(const double *positions, const double *velocities, double box, double dt, long steps, 
        double cutoff, double *potential, double *pressure, long n) {
    // Use d=3 as it's most common in MD
    long d = 3;
    
    // Allocate memory for positions, velocities, and forces
    double *pos = (double *)malloc(n * d * sizeof(double));
    double *vel = (double *)malloc(n * d * sizeof(double));
    double *forces = (double *)malloc(n * d * sizeof(double));
    
    // Copy initial positions and velocities
    memcpy(pos, positions, n * d * sizeof(double));
    memcpy(vel, velocities, n * d * sizeof(double));
    
    // Get initial forces
    double pot, vir;
    interactions(pos, box, cutoff, n, d, forces, &pot, &vir);
    
    double potential_sum = 0.0;
    double pressure_sum = 0.0;
    
    long half_steps = steps / 2;
    long count = steps - half_steps;
    
    for (long step = 0; step < steps; step++) {
        // Velocity Verlet: v(t+0.5*dt) = v(t) + 0.5*dt*F(t)/m (assuming m=1)
        for (long i = 0; i < n * d; i++) {
            vel[i] += 0.5 * dt * forces[i];
        }
        
        // r(t+dt) = r(t) + dt*v(t+0.5*dt)
        for (long i = 0; i < n * d; i++) {
            pos[i] += dt * vel[i];
        }
        
        // Get new forces
        interactions(pos, box, cutoff, n, d, forces, &pot, &vir);
        
        // v(t+dt) = v(t+0.5*dt) + 0.5*dt*F(t+dt)/m
        for (long i = 0; i < n * d; i++) {
            vel[i] += 0.5 * dt * forces[i];
        }
        
        // Accumulate statistics for second half
        if (step >= half_steps) {
            double kinetic = 0.0;
            for (long i = 0; i < n * d; i++) {
                kinetic += 0.5 * vel[i] * vel[i];
            }
            
            potential_sum += pot / n;
            pressure_sum += (2.0 * kinetic + vir) / (d * pow(box, d));
        }
    }
    
    *potential = potential_sum / count;
    *pressure = pressure_sum / count;
    
    free(pos);
    free(vel);
    free(forces);
}
