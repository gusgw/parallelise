#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "kernel_d.h"

static void apply_A_local(double *q, const double *v, long m, long lo, long hi,
                          const double *row_above, const double *row_below,
                          int has_above, int has_below)
{
    for (long i = lo; i < hi; i++) {
        for (long j = 0; j < m; j++) {
            long idx = i * m + j;
            double s = 4.0 * v[idx];
            
            if (i > 0) {
                if (i == lo && has_above) {
                    s -= row_above[j];
                } else if (i > lo) {
                    s -= v[idx - m];
                }
            }
            
            if (i < m - 1) {
                if (i == hi - 1 && has_below) {
                    s -= row_below[j];
                } else if (i < hi - 1) {
                    s -= v[idx + m];
                }
            }
            
            if (j > 0) {
                s -= v[idx - 1];
            }
            
            if (j < m - 1) {
                s -= v[idx + 1];
            }
            
            q[idx] = s;
        }
    }
}

void cgsolve(double *x, const double *b, long m, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long n = m * m;
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    
    long row_lo = lo / m;
    long row_hi = hi / m;
    
    double *r = malloc(sizeof(double) * n);
    double *p = malloc(sizeof(double) * n);
    double *q = malloc(sizeof(double) * n);
    double *row_above = malloc(sizeof(double) * m);
    double *row_below = malloc(sizeof(double) * m);
    
    int rank_above = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_below = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Initialize x = 0, r = b, p = r (only my portion)
    for (long k = 0; k < n; k++) {
        x[k] = 0.0;
        r[k] = 0.0;
        p[k] = 0.0;
    }
    
    for (long k = lo; k < hi; k++) {
        r[k] = b[k];
        p[k] = r[k];
    }
    
    double local_rr = 0.0;
    for (long k = lo; k < hi; k++) {
        local_rr += r[k] * r[k];
    }
    double rr;
    MPI_Allreduce(&local_rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (int iter = 0; iter < iters; iter++) {
        // Exchange halo rows for p
        MPI_Sendrecv(&p[row_lo * m], m, MPI_DOUBLE, rank_above, 0,
                     row_above, m, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&p[(row_hi - 1) * m], m, MPI_DOUBLE, rank_below, 0,
                     row_below, m, MPI_DOUBLE, rank_below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Apply A locally
        apply_A_local(q, p, m, row_lo, row_hi, row_above, row_below,
                      rank > 0, rank < size - 1);
        
        // Compute dot(p, q)
        double local_pq = 0.0;
        for (long k = lo; k < hi; k++) {
            local_pq += p[k] * q[k];
        }
        double pq;
        MPI_Allreduce(&local_pq, &pq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double alpha = rr / pq;
        
        // Update x and r
        for (long k = lo; k < hi; k++) {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }
        
        // Compute dot(r, r)
        double local_rr_new = 0.0;
        for (long k = lo; k < hi; k++) {
            local_rr_new += r[k] * r[k];
        }
        double rr_new;
        MPI_Allreduce(&local_rr_new, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double beta = rr_new / rr;
        rr = rr_new;
        
        // Update p
        for (long k = lo; k < hi; k++) {
            p[k] = r[k] + beta * p[k];
        }
    }
    
    // Gather final x to all ranks
    int *recvcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    
    for (int r = 0; r < size; r++) {
        long r_lo = n * r / size;
        long r_hi = n * (r + 1) / size;
        recvcounts[r] = r_hi - r_lo;
        displs[r] = r_lo;
    }
    
    MPI_Allgatherv(&x[lo], hi - lo, MPI_DOUBLE,
                   x, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    free(recvcounts);
    free(displs);
    free(row_above);
    free(row_below);
    free(r);
    free(p);
    free(q);
}
