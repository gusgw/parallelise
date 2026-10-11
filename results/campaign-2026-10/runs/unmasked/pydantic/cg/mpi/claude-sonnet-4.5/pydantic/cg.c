#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/* Conjugate gradient for the 2D Poisson system on an m x m grid (row-major).
 * Parallel implementation: each rank owns rows [lo, hi) and exchanges boundary
 * rows with neighbours before each A application. Dot products are combined
 * with MPI_Allreduce. The full solution is assembled with MPI_Allgatherv. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = m * m;
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;

    /* Allocate local vectors for rows [lo, hi) */
    double *x_local = (double *)malloc(local_n * sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));

    /* Allocate buffers for boundary rows (halo exchange) */
    double *p_above = (double *)malloc(m * sizeof(double));
    double *p_below = (double *)malloc(m * sizeof(double));

    /* Initialize x = 0, r = b, p = r */
    for (long k = 0; k < local_n; k++) {
        x_local[k] = 0.0;
        r_local[k] = b[lo + k];
        p_local[k] = r_local[k];
    }

    /* Compute initial rr = dot(r, r) */
    double rr_local = 0.0;
    for (long k = 0; k < local_n; k++) {
        rr_local += r_local[k] * r_local[k];
    }
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Determine neighbours for halo exchange */
    int rank_above = (lo >= m) ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (hi <= n - m) ? rank + 1 : MPI_PROC_NULL;

    /* CG iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange boundary rows of p with neighbours */
        /* Send my first row up, receive from above into p_above */
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, rank_above, 0,
                     p_above, m, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send my last row down, receive from below into p_below */
        MPI_Sendrecv(p_local + (local_n - m), m, MPI_DOUBLE, rank_below, 0,
                     p_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p: q = A p (5-point stencil) */
        for (long k = 0; k < local_n; k++) {
            long global_idx = lo + k;
            long i = global_idx / m;
            long j = global_idx % m;
            double s = 4.0 * p_local[k];

            /* Neighbour above (i-1, j) */
            if (i > 0) {
                if (k >= m) {
                    s -= p_local[k - m];
                } else {
                    s -= p_above[j];
                }
            }
            /* Neighbour below (i+1, j) */
            if (i < m - 1) {
                if (k < local_n - m) {
                    s -= p_local[k + m];
                } else {
                    s -= p_below[j];
                }
            }
            /* Neighbour left (i, j-1) */
            if (j > 0) {
                s -= p_local[k - 1];
            }
            /* Neighbour right (i, j+1) */
            if (j < m - 1) {
                s -= p_local[k + 1];
            }
            q_local[k] = s;
        }

        /* Compute alpha = rr / dot(p, q) */
        double pq_local = 0.0;
        for (long k = 0; k < local_n; k++) {
            pq_local += p_local[k] * q_local[k];
        }
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* x += alpha * p */
        for (long k = 0; k < local_n; k++) {
            x_local[k] += alpha * p_local[k];
        }

        /* r -= alpha * q */
        for (long k = 0; k < local_n; k++) {
            r_local[k] -= alpha * q_local[k];
        }

        /* Compute rr_new = dot(r, r) */
        double rr_new_local = 0.0;
        for (long k = 0; k < local_n; k++) {
            rr_new_local += r_local[k] * r_local[k];
        }
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        for (long k = 0; k < local_n; k++) {
            p_local[k] = r_local[k] + beta * p_local[k];
        }
    }

    /* Gather the full solution x on all ranks with MPI_Allgatherv */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = (int)(r_hi - r_lo);
        displs[r] = (int)r_lo;
    }
    MPI_Allgatherv(x_local, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    /* Clean up */
    free(recvcounts);
    free(displs);
    free(p_below);
    free(p_above);
    free(q_local);
    free(p_local);
    free(r_local);
    free(x_local);
}
