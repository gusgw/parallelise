#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

#include "ising.h"

/* Known-good routine (validates the harness; never shown to an agent). Every process runs its
 * own chain on the whole lattice, from the same start but with random numbers of its own, and
 * takes its share of the measuring sweeps. The mean over all the chains agrees with the
 * notebook's function within statistical error. */

static double uniform(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
}

void ising(long side, double beta, long equilibration, long sweeps, long seed, double *energy,
           double *magnetisation)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const long mine = sweeps * (rank + 1) / size - sweeps * rank / size;
    const long n = side * side;

    signed char *s = malloc((size_t)n);
    for (long i = 0; i < n; i++)
        s[i] = 1;
    uint64_t state = (uint64_t)seed * 0x100000001B3ULL + (uint64_t)rank;
    double accept[17]; /* exp(-beta * change) for each change from -8 to 8 */
    for (int c = -8; c <= 8; c++)
        accept[c + 8] = exp(-beta * c);

    double sums[2] = {0.0, 0.0};
    for (long sweep = 0; sweep < equilibration + mine; sweep++) {
        for (int colour = 0; colour < 2; colour++)
            for (long r = 0; r < side; r++)
                for (long c = (r + colour) % 2; c < side; c += 2) {
                    long i = r * side + c;
                    int near = s[((r + side - 1) % side) * side + c] + s[((r + 1) % side) * side + c]
                             + s[r * side + (c + side - 1) % side] + s[r * side + (c + 1) % side];
                    if (uniform(&state) < accept[2 * s[i] * near + 8])
                        s[i] = (signed char)-s[i];
                }
        if (sweep >= equilibration) {
            long bonds = 0, total = 0;
            for (long r = 0; r < side; r++)
                for (long c = 0; c < side; c++) {
                    long i = r * side + c;
                    bonds += s[i] * (s[((r + 1) % side) * side + c] + s[r * side + (c + 1) % side]);
                    total += s[i];
                }
            sums[0] -= (double)bonds / (double)n;
            sums[1] += fabs((double)total) / (double)n;
        }
    }
    free(s);

    double all[2];
    MPI_Reduce(sums, all, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        *energy = all[0] / (double)sweeps;
        *magnetisation = all[1] / (double)sweeps;
    }
}
