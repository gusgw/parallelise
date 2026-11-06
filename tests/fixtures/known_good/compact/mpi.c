#include <mpi.h>
#include <stdlib.h>

#include "compact.h"

/* Known-good routine (validates the harness; never shown to an agent). Each process selects
 * the indices in its own block, in order, and process 0 gathers the selections one block
 * after another, so the indices stay in increasing order. */
void compact(const double *x, double threshold, long *indices, long *indices_count, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const long lo = n * rank / size, hi = n * (rank + 1) / size;

    long *mine = malloc(sizeof(long) * (size_t)(hi > lo ? hi - lo : 1));
    int count = 0;
    for (long i = lo; i < hi; i++)
        if (x[i] > threshold)
            mine[count++] = i;

    int *counts = malloc(sizeof(int) * (size_t)size), *starts = malloc(sizeof(int) * (size_t)size);
    MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    long total = 0;
    if (rank == 0)
        for (int r = 0; r < size; r++) {
            starts[r] = (int)total;
            total += counts[r];
        }
    MPI_Gatherv(mine, count, MPI_LONG, indices, counts, starts, MPI_LONG, 0, MPI_COMM_WORLD);
    if (rank == 0)
        *indices_count = total;
    free(mine);
    free(counts);
    free(starts);
}
