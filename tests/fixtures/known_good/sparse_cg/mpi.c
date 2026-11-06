#include <mpi.h>
#include <stdlib.h>

#include "sparse_cg.h"

/* Known-good routine (validates the harness; never shown to an agent). Each process owns a
 * block of rows: it multiplies them by the search direction, which every process holds in
 * full, and the dot products are summed over all the processes. Process 0 gathers the
 * solution at the end. */
void sparse_cg(const int *indptr, const int *indices, const double *data, const double *b,
               double rtol, long max_iter, double *x, long np1, long nnz, long n)
{
    (void)np1;
    (void)nnz;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *counts = malloc(sizeof(int) * (size_t)size), *starts = malloc(sizeof(int) * (size_t)size);
    for (int r = 0; r < size; r++) {
        starts[r] = (int)(n * r / size);
        counts[r] = (int)(n * (r + 1) / size) - starts[r];
    }
    const long lo = starts[rank], hi = lo + counts[rank];
    double *r = malloc(sizeof(double) * (size_t)n), *p = malloc(sizeof(double) * (size_t)n);
    double *q = malloc(sizeof(double) * (size_t)n);
    for (long i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    double local = 0.0, rr;
    for (long i = lo; i < hi; i++)
        local += b[i] * b[i];
    MPI_Allreduce(&local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double limit = rtol * rtol * rr;
    for (long step = 0; step < max_iter && rr > limit; step++) {
        double pq = 0.0, all_pq, rr_new;
        for (long i = lo; i < hi; i++) {
            double s = 0.0;
            for (int j = indptr[i]; j < indptr[i + 1]; j++)
                s += data[j] * p[indices[j]];
            q[i] = s;
            pq += p[i] * s;
        }
        MPI_Allreduce(&pq, &all_pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double alpha = rr / all_pq;
        local = 0.0;
        for (long i = lo; i < hi; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            local += r[i] * r[i];
        }
        MPI_Allreduce(&local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (long i = lo; i < hi; i++)
            p[i] = r[i] + (rr_new / rr) * p[i];
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, p, counts, starts, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        rr = rr_new;
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : (void *)(x + lo), counts[rank], MPI_DOUBLE, x, counts,
                starts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(r);
    free(p);
    free(q);
    free(counts);
    free(starts);
}
