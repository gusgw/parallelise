#include <mpi.h>
#include <stdlib.h>

#include "cg.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Full-size
 * vectors are allocated on every rank but each rank only ever computes on its own rows
 * [lo, hi). Before each application of A the two boundary rows of p are exchanged with
 * the up/down neighbours (MPI_Sendrecv with MPI_PROC_NULL at the edges), the two dot
 * products combine local partials with MPI_Allreduce so every rank gets identical alpha
 * and beta, and a single final MPI_Allgatherv assembles the whole solution on every
 * rank. Collective; MPI already initialised by the caller. */
void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = m * rank / size;
    const long hi = m * (rank + 1) / size;
    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const long n = m * m;

    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);

    double local = 0.0;
    for (long k = lo * m; k < hi * m; k++) {
        x[k] = 0.0;
        r[k] = b[k];
        p[k] = b[k];
        local += b[k] * b[k];
    }
    double rr = 0.0;
    MPI_Allreduce(&local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int it = 0; it < iters; it++) {
        /* refresh the halo rows of p before applying A: send this rank's first owned
         * row up and its last owned row down (single tag: messages are disambiguated
         * by source rank). Edge halos use a valid dummy pointer since the PROC_NULL
         * send/recv is a no-op. */
        double *top_halo = (lo > 0) ? &p[(lo - 1) * m] : p;
        double *bot_halo = (hi < m) ? &p[hi * m] : p;
        MPI_Sendrecv(&p[lo * m], (int)m, MPI_DOUBLE, up, 0,
                     top_halo, (int)m, MPI_DOUBLE, up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&p[(hi - 1) * m], (int)m, MPI_DOUBLE, down, 0,
                     bot_halo, (int)m, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* q = A p on the owned rows; neighbours off the grid edge are zero */
        double pq_local = 0.0;
        for (long i = lo; i < hi; i++)
            for (long j = 0; j < m; j++) {
                const long idx = i * m + j;
                double s = 4.0 * p[idx];
                if (i > 0)     s -= p[idx - m];
                if (i < m - 1) s -= p[idx + m];
                if (j > 0)     s -= p[idx - 1];
                if (j < m - 1) s -= p[idx + 1];
                q[idx] = s;
                pq_local += p[idx] * s;
            }
        double pq = 0.0;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        const double alpha = rr / pq;
        double rr_local = 0.0;
        for (long k = lo * m; k < hi * m; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
            rr_local += r[k] * r[k];
        }
        double rr_new = 0.0;
        MPI_Allreduce(&rr_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        const double beta = rr_new / rr;
        rr = rr_new;
        for (long k = lo * m; k < hi * m; k++)
            p[k] = r[k] + beta * p[k];
    }

    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    for (int rk = 0; rk < size; rk++) {
        long rl = m * rk / size, rh = m * (rk + 1) / size;
        counts[rk] = (int)((rh - rl) * m);
        displs[rk] = (int)(rl * m);
    }
    /* each rank's block is already at displs[rank] in x -> gather in place */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    free(counts);
    free(displs);
    free(r);
    free(p);
    free(q);
}
