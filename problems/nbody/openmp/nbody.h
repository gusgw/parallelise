#ifndef NBODY_H
#define NBODY_H

/* Direct 2D N-body acceleration step with softening. Body i has unit mass and sits at
 * (px[i], py[i]); the kernel computes, for every i, the direct sum over all j != i:
 *     ax[i] = sum of dx / d3,   ay[i] = sum of dy / d3
 * where dx = px[j] - px[i], dy = py[j] - py[i], r2 = dx*dx + dy*dy, and
 * d3 = (r2 + EPS2) * sqrt(r2 + EPS2) with EPS2 = 0.0025 (softening 0.05 squared).
 *
 * Implement THIS EXACT signature in nbody.c. Parallelise with OpenMP (e.g. a
 * '#pragma omp parallel for' over the bodies i; each body's accelerations are
 * independent of the others'). Plain shared-memory function: no MPI. */
void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n);

#endif /* NBODY_H */
