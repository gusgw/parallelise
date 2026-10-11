#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cg.h"
#include "problem.h"

/* Helper to compute dot product of local vectors */
static double local_dot(const double *a, const double *b, long n)
{
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine row ownership */
    long lo = m * rank / size;
    long hi = m * (rank + 1) / size;
    long local_rows = hi - lo;
    long local_n = local_rows * m; /* number of elements owned */

    /* Allocate local vectors */
    double *x_loc = (double *)calloc(local_n, sizeof(double));
    double *r = (double *)malloc(local_n * sizeof(double));
    double *p = (double *)malloc(local_n * sizeof(double));
    double *q = (double *)malloc(local_n * sizeof(double));
    if (!x_loc || !r || !p || !q) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Initialize r = b_local, p = r, x_loc already zero */
    for (long i = 0; i < local_n; ++i) {
        r[i] = b[lo * m + i];
        p[i] = r[i];
    }

    /* Initial rr = dot(r,r) */
    double rr_local = local_dot(r, r, local_n);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Buffers for halo exchange */
    int up = rank - 1;
    int down = rank + 1;
    if (up < 0) up = MPI_PROC_NULL;
    if (down >= size) down = MPI_PROC_NULL;

    double *p_ext = (double *)malloc((local_rows + 2) * m * sizeof(double));
    if (!p_ext) MPI_Abort(MPI_COMM_WORLD, 1);

    for (int it = 0; it < iters; ++it) {
        /* Prepare extended p with halos */
        memcpy(p_ext + m, p, local_n * sizeof(double));
        if (up == MPI_PROC_NULL) {
            memset(p_ext, 0, m * sizeof(double));
        }
        if (down == MPI_PROC_NULL) {
            memset(p_ext + (local_rows + 1) * m, 0, m * sizeof(double));
        }
        /* Exchange boundary rows */
        MPI_Sendrecv(p, m, MPI_DOUBLE, up, 0,
                     p_ext + (local_rows + 1) * m, m, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(p + (local_rows - 1) * m, m, MPI_DOUBLE, down, 1,
                     p_ext, m, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to p -> q */
        for (long i = 0; i < local_rows; ++i) {
            for (long j = 0; j < m; ++j) {
                double s = 4.0 * p_ext[(i + 1) * m + j];
                /* up neighbor */
                s -= p_ext[i * m + j];
                /* down neighbor */
                s -= p_ext[(i + 2) * m + j];
                /* left neighbor */
                if (j > 0) s -= p_ext[(i + 1) * m + (j - 1)];
                /* right neighbor */
                if (j < m - 1) s -= p_ext[(i + 1) * m + (j + 1)];
                q[i * m + j] = s;
            }
        }

        /* Compute alpha = rr / dot(p,q) */
        double pq_local = local_dot(p, q, local_n);
        double pq;
        MPI_Allreduce(&pq_local, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = rr / pq;

        /* Update x_loc and r */
        for (long i = 0; i < local_n; ++i) {
            x_loc[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        /* Compute new rr */
        double rr_new_local = local_dot(r, r, local_n);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        /* Update p = r + beta * p */
        for (long i = 0; i < local_n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    /* Gather full solution x on all ranks */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        long l_lo = m * i / size;
        long l_hi = m * (i + 1) / size;
        recvcounts[i] = (int)((l_hi - l_lo) * m);
        displs[i] = (int)(l_lo * m);
    }
    MPI_Allgatherv(x_loc, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(x_loc);
    free(r);
    free(p);
    free(q);
    free(p_ext);
    free(recvcounts);
    free(displs);
}
