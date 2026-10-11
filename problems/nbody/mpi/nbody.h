#ifndef NBODY_H
#define NBODY_H

/* Direct 2D N-body acceleration step with softening. Body i has unit mass and sits at
 * (px[i], py[i]); the kernel computes, for every i, the direct sum over all j != i:
 *     ax[i] = sum of dx / d3,   ay[i] = sum of dy / d3
 * where dx = px[j] - px[i], dy = py[j] - py[i], r2 = dx*dx + dy*dy, and
 * d3 = (r2 + EPS2) * sqrt(r2 + EPS2) with EPS2 = 0.0025 (softening 0.05 squared).
 *
 * Implement THIS EXACT signature in nbody.c. It is a COLLECTIVE routine: every MPI
 * rank calls it together with identical px and py (each rank holds full copies of all
 * the positions), and on return every rank holds the full ax and ay. Assume MPI is
 * already initialised by the caller (main); do NOT call MPI_Init or MPI_Finalize
 * inside it. Decompose the bodies i across the ranks (each rank computes the
 * accelerations only for its block of i; no halo exchange is needed because the
 * positions are read-only and already everywhere) and assemble the full ax and ay on
 * every rank with a gather (MPI_Allgather / MPI_Allgatherv). */
void nbody_accel(const double *px, const double *py, double *ax, double *ay, long n);

#endif /* NBODY_H */
