#include <omp.h>
#include <stdlib.h>

#include "reservoir.h"
#include "problem.h"   /* RES_CAP */

/* Reference solution (validates the harness -- NOT shown to the agents).
 *
 * One step is level' = min(CAP, max(0, level + a[i])), a "shift then clamp" of the level.
 * Any composition of such steps is again a shift-then-clamp, so a block is summarised by
 * three numbers (d, lo, hi) standing for the map
 *     f(r) = min(hi, max(lo, r + d)).
 * A single step i is (d, lo, hi) = (a[i], 0, CAP). Two maps compose (apply g, then f):
 *     (f o g).d  = g.d + f.d
 *     (f o g).lo = clamp(g.lo + f.d, f.lo, f.hi)
 *     (f o g).hi = clamp(g.hi + f.d, f.lo, f.hi)
 * with clamp(x, l, h) = min(h, max(l, x)). This composition is associative, so the
 * classic two-pass block scan works with (d, lo, hi) as the block summary. It is NOT a
 * prefix sum: the clamps make the map nonlinear, and dropping them gives wrong answers. */

typedef struct { double d, lo, hi; } Clamp;

static inline double clampd(double x, double lo, double hi)
{
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

static inline double apply_clamp(const Clamp *f, double r)
{
    return clampd(r + f->d, f->lo, f->hi);
}

/* h = f o g  (do g first, then f). */
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
    if (n <= 0)
        return;
    Clamp *block = NULL;    /* one composed map per thread */
    double *enter = NULL;   /* the level entering each thread's block */
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        #pragma omp single
        {
            block = malloc(sizeof(Clamp) * (size_t)nt);
            enter = malloc(sizeof(double) * (size_t)nt);
        }
        /* the single's implicit barrier makes block and enter visible to every thread */

        const long lo = n * tid / nt;
        const long hi = n * (tid + 1) / nt;

        /* pass 1: fold this thread's block into one shift-then-clamp map. The identity is
         * a shift of 0 with limits wide open. */
        Clamp acc = {0.0, -1.0e18, 1.0e18};
        for (long i = lo; i < hi; i++) {
            Clamp f = {a[i], 0.0, RES_CAP};
            acc = compose(&f, &acc);
        }
        block[tid] = acc;
        #pragma omp barrier

        /* apply the thread maps in order to the initial level (0) to get each block's
         * entering level */
        #pragma omp single
        {
            double level = 0.0;
            for (int t = 0; t < nt; t++) {
                enter[t] = level;
                level = apply_clamp(&block[t], level);
            }
        }
        /* the single's implicit barrier orders the entry levels before pass 2 */

        /* pass 2: run the plain recurrence over this block from its entering level */
        double level = enter[tid];
        for (long i = lo; i < hi; i++) {
            level = clampd(level + a[i], 0.0, RES_CAP);
            out[i] = level;
        }
    }
    free(block);
    free(enter);
}
