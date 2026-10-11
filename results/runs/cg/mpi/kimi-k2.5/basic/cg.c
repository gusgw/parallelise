#include "cg.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Apply the 5-point Poisson stencil to local rows of p.
 * Each rank owns rows [lo, hi) of the m x m grid.
 * Before calling, p must have halo rows exchanged (ghost rows at lo-1 and hi
 * from neighboring ranks, or zero at boundaries). */
static void apply_A_local(double *q_local, const double *p_with_halo,
                          long m, long lo, long hi)
{
    /* p_with_halo contains rows lo-1 to hi (inclusive), so local row i
     * corresponds to p_with_halo[(i - lo + 1) * m + j] */
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx_local = (i - lo) * m + j;
            long idx_halo = (i - lo + 1) * m + j;
            double s = 4.0 * p_with_halo[idx_halo];
            /* up neighbor */
            s -= p_with_halo[idx_halo - m];
            /* down neighbor */
            s -= p_with_halo[idx_halo + m];
            /* left neighbor */
            if (j > 0) s -= p_with_halo[idx_halo - 1];
            /* right neighbor */
            if (j < m - 1) s -= p_with_halo[idx_halo + 1];
            q_local[idx_local] = s;
        }
    }
}

/* Compute local dot product for rows [lo, hi) */
static double dot_local(const double *u, const double *v, long m, long lo, long hi)
{
    double s = 0.0;
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = (i - lo) * m + j;
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
    
    /* Decompose rows across ranks: each rank owns rows [lo, hi) */
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    /* Convert to row indices (each row has m elements) */
    long row_lo = lo / m;
    long row_hi = hi / m;
    /* Ensure we have complete rows */
    if (lo % m != 0) row_lo++;
    if (hi % m != 0) row_hi++;
    /* Recalculate lo, hi based on complete rows */
    lo = row_lo * m;
    hi = row_hi * m;
    
    long local_rows = row_hi - row_lo;
    long local_n = local_rows * m;

    /* Allocate local arrays */
    double *x_local = (double *)malloc(local_n * sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));
    /* p_with_halo has local rows plus up to 2 halo rows (one above, one below) */
    double *p_with_halo = (double *)malloc((local_rows + 2) * m * sizeof(double));

    /* Initialize: x = 0, r = b, p = r */
    for (long i = 0; i < local_n; i++) {
        x_local[i] = 0.0;
        r_local[i] = b[lo + i];
        p_local[i] = r_local[i];
    }

    /* Compute initial rr = dot(r, r) */
    double rr_local = dot_local(r_local, r_local, m, row_lo, row_hi);
    double rr;
    MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows of p before applying A */
        /* Copy local p to the middle of p_with_halo */
        memcpy(p_with_halo + m, p_local, local_n * sizeof(double));
        
        /* Determine neighbors */
        int up_rank = (row_lo > 0) ? rank - 1 : MPI_PROC_NULL;
        int down_rank = (row_hi < m) ? rank + 1 : MPI_PROC_NULL;
        
        /* Exchange with up neighbor (send first row, receive halo row above) */
        MPI_Sendrecv(p_local, m, MPI_DOUBLE, up_rank, 0,
                     p_with_halo, m, MPI_DOUBLE, up_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Exchange with down neighbor (send last row, receive halo row below) */
        MPI_Sendrecv(p_local + (local_rows - 1) * m, m, MPI_DOUBLE, down_rank, 1,
                     p_with_halo + (local_rows + 1) * m, m, MPI_DOUBLE, down_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Apply A to local rows */
        apply_A_local(q_local, p_with_halo, m, row_lo, row_hi);

        /* Compute dot(p, q) */
        double pdotq_local = dot_local(p_local, q_local, m, row_lo, row_hi);
        double pdotq;
        MPI_Allreduce(&pdotq_local, &pdotq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* alpha = rr / dot(p, q) */
        double alpha = rr / pdotq;

        /* x += alpha * p, r -= alpha * q */
        for (long i = 0; i < local_n; i++) {
            x_local[i] += alpha * p_local[i];
            r_local[i] -= alpha * q_local[i];
        }

        /* Compute new rr = dot(r, r) */
        double rr_new_local = dot_local(r_local, r_local, m, row_lo, row_hi);
        double rr_new;
        MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* beta = rr_new / rr */
        double beta = rr_new / rr;

        /* p = r + beta * p */
        for (long i = 0; i < local_n; i++) {
            p_local[i] = r_local[i] + beta * p_local[i];
        }

        rr = rr_new;
    }

    /* Gather the full solution x on all ranks using Allgatherv */
    /* First, compute counts and displacements */
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    /* Each rank reports its local_n */
    int local_n_int = (int)local_n;
    MPI_Allgather(&local_n_int, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    /* Compute displacements */
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }

    /* Allgatherv to get full x on all ranks */
    MPI_Allgatherv(x_local, local_n_int, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Cleanup */
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
    free(p_with_halo);
    free(recvcounts);
    free(displs);
}
