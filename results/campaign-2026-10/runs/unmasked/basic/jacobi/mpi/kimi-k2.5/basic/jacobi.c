#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Calculate row range for this rank (interior rows only, excluding boundaries) */
    long interior_rows = n - 2;  /* rows 1 to n-2 are interior */
    long rows_per_rank = interior_rows / size;
    long remainder = interior_rows % size;
    
    /* Each rank gets rows_per_rank or rows_per_rank+1 interior rows */
    long local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    long row_start = 1 + rank * rows_per_rank + (rank < remainder ? rank : remainder);

    /* Allocate local buffers with halo rows */
    /* local_u holds rows [row_start-1, row_end) i.e., local_rows + 2 rows total */
    long local_total_rows = local_rows + 2;
    double *local_u = malloc(sizeof(double) * local_total_rows * n);
    double *local_un = malloc(sizeof(double) * local_total_rows * n);
    
    if (!local_u || !local_un) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    /* Initialize local buffers from global u */
    for (long i = 0; i < local_total_rows; i++) {
        long global_i = row_start - 1 + i;
        memcpy(&local_u[i * n], &u[global_i * n], sizeof(double) * n);
        memcpy(&local_un[i * n], &u[global_i * n], sizeof(double) * n);
    }

    /* Determine neighbors for halo exchange */
    int up_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Exchange halo rows with neighbors */
        /* Send top interior row (row 1) up, receive into halo row 0 */
        MPI_Sendrecv(&local_u[1 * n], n, MPI_DOUBLE, up_neighbor, 0,
                     &local_u[0 * n], n, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /* Send bottom interior row (row local_total_rows-2) down, receive into halo row local_total_rows-1 */
        MPI_Sendrecv(&local_u[(local_total_rows - 2) * n], n, MPI_DOUBLE, down_neighbor, 1,
                     &local_u[(local_total_rows - 1) * n], n, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Update interior cells (local rows 1 to local_total_rows-2 correspond to global row_start to row_end-1) */
        for (long li = 1; li <= local_rows; li++) {
            for (long j = 1; j < n - 1; j++) {
                local_un[li * n + j] = 0.25 * (
                    local_u[(li - 1) * n + j] +
                    local_u[(li + 1) * n + j] +
                    local_u[li * n + (j - 1)] +
                    local_u[li * n + (j + 1)]
                );
            }
        }

        /* Swap buffers */
        double *tmp = local_u;
        local_u = local_un;
        local_un = tmp;
    }

    /* Gather results from all ranks to reconstruct full grid on all ranks */
    /* Each rank contributes its local interior rows (excluding halos) */
    /* Use MPI_Allgatherv to handle non-uniform row distribution */
    
    /* First, gather the counts (number of elements each rank has) */
    int *recvcounts = NULL;
    int *displs = NULL;
    
    recvcounts = malloc(sizeof(int) * size);
    displs = malloc(sizeof(int) * size);
    
    if (!recvcounts || !displs) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    int my_count = (int)(local_rows * n);
    MPI_Allgather(&my_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    
    /* Compute displacements (in elements) */
    int disp = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = disp;
        disp += recvcounts[i];
    }
    
    /* Allocate temporary buffer for gathering */
    double *gathered = malloc(sizeof(double) * interior_rows * n);
    if (!gathered) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    /* Each rank contributes its interior rows (skip halo rows) */
    /* local_u[1..local_rows] are the interior rows */
    MPI_Allgatherv(&local_u[1 * n], (int)(local_rows * n), MPI_DOUBLE,
                   gathered, recvcounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    /* Copy gathered data back to u (only interior rows, boundaries stay fixed) */
    for (long i = 0; i < interior_rows; i++) {
        memcpy(&u[(i + 1) * n], &gathered[i * n], sizeof(double) * n);
    }
    
    /* Cleanup */
    free(local_u);
    free(local_un);
    free(gathered);
    free(recvcounts);
    free(displs);
}
