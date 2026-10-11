#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Apply the 5-point Poisson stencil to local rows of p.
 * Each rank owns rows [row_lo, row_hi) of the m x m grid.
 * Before calling, p must include halo rows from neighbors.
 * The result q is stored for local rows only.
 */
static void apply_A_local(double *q, const double *p, long m, long nlocal_rows)
{
    for (long i = 0; i < nlocal_rows; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;       /* index in local array (no halo) */
            long pidx = (i + 1) * m + j; /* index in p with halo (row i+1 in p) */
            double s = 4.0 * p[pidx];
            /* up neighbor */
            s -= p[pidx - m];
            /* down neighbor */
            s -= p[pidx + m];
            /* left neighbor */
            if (j > 0) s -= p[pidx - 1];
            /* right neighbor */
            if (j < m - 1) s -= p[pidx + 1];
            q[idx] = s;
        }
    }
}

/* Compute local dot product for rows [row_lo, row_hi) */
static double dot_local(const double *u, const double *v, long m, long row_lo, long row_hi)
{
    double s = 0.0;
    for (long i = row_lo; i < row_hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            s += u[idx] * v[idx];
        }
    }
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = m * m;

    /* Decompose rows across ranks: each rank owns rows [row_lo, row_hi) */
    long row_lo = m * rank / size;
    long row_hi = m * (rank + 1) / size;
    long nlocal_rows = row_hi - row_lo;
    long nlocal = nlocal_rows * m;  /* number of elements this rank owns */

    /* Allocate local arrays */
    /* p needs halo rows: one above and one below */
    double *p = (double *)calloc((nlocal_rows + 2) * m, sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));
    double *q = (double *)malloc(nlocal * sizeof(double));

    /* Initialize: x = 0, r = b, p = r */
    /* x is already zero on entry (caller responsibility) */
    memcpy(r, b, n * sizeof(double));

    /* Copy r to p for local rows (p has halo, so offset by m) */
    for (long i = row_lo; i < row_hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            long pidx = (i - row_lo + 1) * m + j;  /* +1 for halo */
            p[pidx] = r[idx];
        }
    }

    /* rr = dot(r, r) */
    double rr_local = dot_local(r, r, m, row_lo, row_hi);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows of p with neighbors */
        
        int up = rank - 1;
        int down = rank + 1;
        if (up < 0) up = MPI_PROC_NULL;
        if (down >= size) down = MPI_PROC_NULL;

        /* Send top row (row_lo) up, receive from down into halo at bottom */
        MPI_Sendrecv(&p[m],          /* send: first real row */
                     m, MPI_DOUBLE,
                     up, 0,
                     &p[(nlocal_rows + 1) * m], /* recv: bottom halo */
                     m, MPI_DOUBLE,
                     down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send bottom row (row_hi-1) down, receive from up into halo at top */
        MPI_Sendrecv(&p[nlocal_rows * m], /* send: last real row */
                     m, MPI_DOUBLE,
                     down, 1,
                     &p[0],          /* recv: top halo */
                     m, MPI_DOUBLE,
                     up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* q = A * p */
        apply_A_local(q, p, m, nlocal_rows);

        /* pdotq = dot(p, q) - only for local rows */
        double pdotq_local = 0.0;
        for (long i = 0; i < nlocal_rows; i++) {
            for (long j = 0; j < m; j++) {
                long pidx = (i + 1) * m + j;
                long qidx = i * m + j;
                pdotq_local += p[pidx] * q[qidx];
            }
        }
        double pdotq;
        MPI_Allreduce(&pdotq_local, &pdotq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* alpha = rr / pdotq */
        double alpha = rr / pdotq;

        /* x += alpha * p */
        for (long i = row_lo; i < row_hi; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                long pidx = (i - row_lo + 1) * m + j;
                x[idx] += alpha * p[pidx];
            }
        }

        /* r -= alpha * q */
        for (long i = row_lo; i < row_hi; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                long qidx = (i - row_lo) * m + j;
                r[idx] -= alpha * q[qidx];
            }
        }

        /* rr_new = dot(r, r) */
        double rr_new_local = dot_local(r, r, m, row_lo, row_hi);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;
        rr = rr_new;

        /* p = r + beta * p */
        for (long i = row_lo; i < row_hi; i++) {
            for (long j = 0; j < m; j++) {
                long idx = i * m + j;
                long pidx = (i - row_lo + 1) * m + j;
                p[pidx] = r[idx] + beta * p[pidx];
            }
        }
    }

    /* Gather full x on all ranks using Allgatherv */
    /* First, compute counts and displacements (in elements) */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        long r_row_lo = m * i / size;
        long r_row_hi = m * (i + 1) / size;
        recvcounts[i] = (int)((r_row_hi - r_row_lo) * m);
        displs[i] = (int)(r_row_lo * m);
    }

    /* Create a temporary buffer for the local portion */
    double *xlocal = (double *)malloc(nlocal * sizeof(double));
    for (long i = row_lo; i < row_hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            long local_idx = (i - row_lo) * m + j;
            xlocal[local_idx] = x[idx];
        }
    }

    MPI_Allgatherv(xlocal, (int)nlocal, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(p);
    free(r);
    free(q);
    free(recvcounts);
    free(displs);
    free(xlocal);
}
