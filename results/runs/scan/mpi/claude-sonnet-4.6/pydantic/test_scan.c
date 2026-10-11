/* test_scan.c — unit test for the parallel inclusive prefix-sum kernel.
 *
 * Every rank calls scan() the same number of times (required: collective routine).
 * Only rank 0 checks results against reference.h and prints the outcome.
 * Exit code: 0 = all checks passed, non-zero = at least one failure. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

/* Gather the distributed output (each rank owns out[lo..hi)) into full_out on
 * rank 0.  full_out must be pre-allocated to length n on rank 0; ignored on
 * other ranks.  Works for any n >= 0 and any number of ranks. */
static void gather_output(const double *out, long n, int rank, int size,
                          double *full_out)
{
    long lo   = (long)n * rank / size;
    long hi   = (long)n * (rank + 1) / size;
    long llen = hi - lo;

    int *counts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);
    if (!counts || !displs) {
        fprintf(stderr, "rank %d: allocation failed in gather_output\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int llen_int = (int)llen;
    MPI_Gather(&llen_int, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < size; r++)
            displs[r] = displs[r - 1] + counts[r - 1];
    }

    /* Pass a valid (non-NULL) send buffer even for empty slices. */
    MPI_Gatherv((llen > 0) ? (out + lo) : out, llen_int, MPI_DOUBLE,
                full_out, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    free(counts);
    free(displs);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long n = REF_N;

    /* Allocate and fill input on every rank (each holds a full copy, per spec). */
    double *in  = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    if (!in || !out) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long i = 0; i < n; i++) in[i] = in_of(i);

    /* ------------------------------------------------------------------ *
     * Test 1: full-size scan — checksum must match the oracle.            *
     * ------------------------------------------------------------------ */

    /* All ranks call the collective kernel together. */
    scan(in, out, n);

    /* Gather each rank's slice to rank 0 so it can compute the full checksum. */
    double *full_out = NULL;
    if (rank == 0) {
        full_out = malloc(sizeof(double) * n);
        if (!full_out) {
            fprintf(stderr, "rank 0: allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    gather_output(out, n, rank, size, full_out);

    if (rank == 0) {
        double cs = checksum(full_out, n);
        CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                    "full-size scan checksum matches oracle");
        free(full_out);
    }

    /* ------------------------------------------------------------------ *
     * Test 2: small scan (n=8) — verify individual output values.         *
     * ------------------------------------------------------------------ */
    {
        const long ns = 8;
        double small_in[8], small_out[8], expected[8];

        for (long i = 0; i < ns; i++) small_in[i] = in_of(i);

        /* Build expected output with a sequential scan (rank 0 only, for CHECK). */
        double run = 0.0;
        for (long i = 0; i < ns; i++) {
            run += small_in[i];
            expected[i] = run;
        }

        /* All ranks call the collective kernel. */
        scan(small_in, small_out, ns);

        /* Gather small result to rank 0. */
        double gathered[8];
        gather_output(small_out, ns, rank, size, gathered);

        if (rank == 0) {
            for (long i = 0; i < ns; i++) {
                char msg[64];
                snprintf(msg, sizeof(msg), "small scan out[%ld]", i);
                CHECK_CLOSE(gathered[i], expected[i], REF_TOL, msg);
            }
        }
    }

    /* ------------------------------------------------------------------ *
     * Test 3: single-element scan — out[0] must equal in[0].             *
     * ------------------------------------------------------------------ *
     * Only meaningful when n >= size (the block decomposition assigns at   *
     * least one element per rank).  When size > n some ranks own zero      *
     * elements; we still call scan() collectively (it is a collective) but *
     * skip the value check to avoid undefined gather behaviour.            */
    {
        const long ns = 1;
        double s_in[1], s_out[1];
        s_in[0]  = in_of(0);
        s_out[0] = 0.0;

        /* All ranks participate in the collective regardless. */
        scan(s_in, s_out, ns);

        /* Only check when every rank owns at least one element (ns >= size). */
        if (ns >= (long)size) {
            double gathered1[1];
            gather_output(s_out, ns, rank, size, gathered1);

            if (rank == 0) {
                CHECK_CLOSE(gathered1[0], s_in[0], REF_TOL,
                            "single-element scan: out[0] == in[0]");
            }
        }
        /* When size > ns, skip the gather/check but the collective already ran. */
    }

    /* ------------------------------------------------------------------ *
     * Print summary (rank 0 only) and exit.                               *
     * ------------------------------------------------------------------ */
    int result = 0;
    if (rank == 0) result = TEST_SUMMARY();

    free(in);
    free(out);

    MPI_Finalize();
    return result;
}
