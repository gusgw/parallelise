#include <mpi.h>
#include <stdlib.h>
#include "scan.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Block-scan
 * decomposition: each rank scans its contiguous block [lo, hi) of in locally, an
 * gather of the block totals from every rank gives each rank its starting offset (the
 * sum of the totals of all lower ranks, so 0 on rank 0), and the offset is added to the
 * block. Process 0 then gathers the blocks. Collective; assumes MPI already initialised. */
void scan(const double *in, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;

    /* local inclusive scan of the owned block; total is the block's sum */
    double total = 0.0;
    for (long i = lo; i < hi; i++) {
        total += in[i];
        out[i] = total;
    }

    /* gather every rank's block total, then sum the totals of the lower ranks to get
     * this rank's starting offset. Only one double per rank is exchanged. */
    double *totals = malloc((size_t)size * sizeof *totals);
    if (totals == NULL) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Allgather(&total, 1, MPI_DOUBLE, totals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double offset = 0.0;
    for (int r = 0; r < rank; r++) offset += totals[r];
    free(totals);
    for (long i = lo; i < hi; i++) out[i] += offset;
    /* Process 0 must hold the whole result, so every block is gathered to it. */
    int *counts = malloc(sizeof(int) * (size_t)size), *starts = malloc(sizeof(int) * (size_t)size);
    for (int r = 0; r < size; r++) {
        starts[r] = (int)(n * r / size);
        counts[r] = (int)(n * (r + 1) / size - starts[r]);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : (void *)(out + lo), (int)(hi - lo), MPI_DOUBLE, out,
                counts, starts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(counts);
    free(starts);
}
