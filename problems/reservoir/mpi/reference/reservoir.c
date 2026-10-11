#include <mpi.h>
#include <stdlib.h>

#include "reservoir.h"
#include "problem.h"   /* RES_CAP */

/* Reference solution (validates the harness -- NOT shown to the agents).
 *
 * One step is level' = min(CAP, max(0, level + a[i])), a "shift then clamp". Any
 * composition of steps is again a shift-then-clamp, summarised by three numbers
 * (d, lo, hi) standing for f(r) = min(hi, max(lo, r + d)). A single step i is
 * (a[i], 0, CAP), and two maps compose (apply g, then f) by
 *     (f o g).d  = g.d + f.d
 *     (f o g).lo = clamp(g.lo + f.d, f.lo, f.hi)
 *     (f o g).hi = clamp(g.hi + f.d, f.lo, f.hi).
 * So each rank folds its own block into one such map, the ranks exchange their maps
 * (there are only `size` of them), and each rank composes the maps of all lower ranks --
 * applied to the initial level 0 -- to get the level entering its block. It then runs the
 * plain recurrence over its block. The result stays distributed per the contract. It is
 * NOT a prefix sum: the clamps make the map nonlinear. */

typedef struct { double d, lo, hi; } Clamp;   /* 3 contiguous doubles */

static inline double clampd(double x, double lo, double hi)
{
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

/* h = f o g (do g first, then f). */
static inline Clamp compose(const Clamp *f, const Clamp *g)
{
    Clamp h;
    h.d = g->d + f->d;
    h.lo = clampd(g->lo + f->d, f->lo, f->hi);
    h.hi = clampd(g->hi + f->d, f->lo, f->hi);
    return h;
}

void reservoir(const double *a, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;

    /* fold this rank's block into one shift-then-clamp map (identity if empty) */
    Clamp acc = {0.0, -1.0e18, 1.0e18};
    for (long i = lo; i < hi; i++) {
        Clamp f = {a[i], 0.0, RES_CAP};
        acc = compose(&f, &acc);
    }

    /* exchange the per-rank maps (3 doubles each) so every rank has all of them */
    Clamp *all = malloc(sizeof(Clamp) * (size_t)size);
    MPI_Allgather(&acc, 3, MPI_DOUBLE, all, 3, MPI_DOUBLE, MPI_COMM_WORLD);

    /* compose the maps of all lower ranks, applied to the initial level 0, to get the
     * level entering this rank's block */
    double level = 0.0;
    for (int t = 0; t < rank; t++)
        level = clampd(level + all[t].d, all[t].lo, all[t].hi);
    free(all);

    /* run the plain recurrence over this rank's block from its entering level */
    for (long i = lo; i < hi; i++) {
        level = clampd(level + a[i], 0.0, RES_CAP);
        out[i] = level;
    }
    /* the result stays distributed: this rank's block of out is correct -- no gather */
}
