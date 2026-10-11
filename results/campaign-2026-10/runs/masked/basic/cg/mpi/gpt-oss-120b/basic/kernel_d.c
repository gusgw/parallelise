#include "kernel_d.h"
#include "problem.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Apply the 5‑point Laplacian A to vector p (distributed rows).
   The result q is written into the provided buffer (same local size as p).
   Halo rows are exchanged with neighbouring ranks.
*/
static void apply_A_local(const double *p_local, double *q_local,
                          long m, long local_rows,
                          int rank, int size)
{
    if (local_rows == 0) return; // nothing to do
    // Allocate extended array with two halo rows
    double *p_ext = (double *)malloc((local_rows + 2) * m * sizeof(double));
    // interior copy
    memcpy(p_ext + m, p_local, local_rows * m * sizeof(double));
    // Initialize halo rows (will be overwritten by exchange)
    memset(p_ext, 0, m * sizeof(double));
    memset(p_ext + (local_rows + 1) * m, 0, m * sizeof(double));

    // Determine neighbour ranks, using MPI_PROC_NULL for non‑existent neighbours
    int up   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    MPI_Status status;
    // Send first interior row upward, receive into top halo
    MPI_Sendrecv(p_ext + m, m, MPI_DOUBLE, up, 0,
                 p_ext, m, MPI_DOUBLE, up, 0,
                 MPI_COMM_WORLD, &status);
    // Send last interior row downward, receive into bottom halo
    MPI_Sendrecv(p_ext + (local_rows) * m, m, MPI_DOUBLE, down, 0,
                 p_ext + (local_rows + 1) * m, m, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, &status);

    // Compute q = A * p for interior rows
    for (long i = 0; i < local_rows; ++i) {
        for (long j = 0; j < m; ++j) {
            double s = 4.0 * p_ext[(i + 1) * m + j];
            // up neighbor
            s -= p_ext[i * m + j];
            // down neighbor
            s -= p_ext[(i + 2) * m + j];
            // left neighbor
            if (j > 0) s -= p_ext[(i + 1) * m + (j - 1)];
            // right neighbor
            if (j < m - 1) s -= p_ext[(i + 1) * m + (j + 1)];
            q_local[i * m + j] = s;
        }
    }
    free(p_ext);
}

static double dot_local(const double *a, const double *b, long n)
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

    long rows_start = (m * rank) / size;
    long rows_end   = (m * (rank + 1)) / size;
    long local_rows = rows_end - rows_start;
    long local_n    = local_rows * m; // number of elements owned by this rank

    // Allocate local vectors
    double *x_local = (double *)calloc(local_n, sizeof(double));
    double *r_local = (double *)malloc(local_n * sizeof(double));
    double *p_local = (double *)malloc(local_n * sizeof(double));
    double *q_local = (double *)malloc(local_n * sizeof(double));

    // Fill local copy of b and initialise r = b, p = r
    for (long i = 0; i < local_rows; ++i) {
        long global_i = rows_start + i;
        for (long j = 0; j < m; ++j) {
            long idx = global_i * m + j;
            double val = b[idx];
            r_local[i * m + j] = val;
            p_local[i * m + j] = val;
        }
    }

    // Initial residual norm rr = r·r (global reduction)
    double local_rr = dot_local(r_local, r_local, local_n);
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int it = 0; it < iters; ++it) {
        // q = A * p (distributed)
        apply_A_local(p_local, q_local, m, local_rows, rank, size);

        // Compute p·q globally
        double local_pq = dot_local(p_local, q_local, local_n);
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rr / pq;

        // x = x + alpha * p   (local update)
        for (long i = 0; i < local_n; ++i) x_local[i] += alpha * p_local[i];
        // r = r - alpha * q   (local update)
        for (long i = 0; i < local_n; ++i) r_local[i] -= alpha * q_local[i];

        // Compute new residual norm
        double local_rr_new = dot_local(r_local, r_local, local_n);
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rr_new / rr;
        rr = rr_new;

        // p = r + beta * p   (local update)
        for (long i = 0; i < local_n; ++i) p_local[i] = r_local[i] + beta * p_local[i];
    }

    // Gather the full solution vector onto all ranks (so that caller sees complete x)
    // Build counts and displacements for Allgatherv
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        long rs = (m * i) / size;
        long re = (m * (i + 1)) / size;
        recvcounts[i] = (int)((re - rs) * m);
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];

    MPI_Allgatherv(x_local, (int)local_n, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
    free(x_local);
    free(r_local);
    free(p_local);
    free(q_local);
}
