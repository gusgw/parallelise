#include <omp.h>
#include <stdlib.h>

#include "decay.h"
#include "problem.h"   /* DECAY_A, DECAY_B, DECAY_NEG */

/* Reference solution (validates the harness -- NOT shown to the agents).
 *
 * The recurrence out[i] = max(A + out[i-1], B + out[i-2], w[i]) carries the pair of
 * most-recent scores s = (out[i-1], out[i-2]). Step i turns s into (out[i], out[i-1]) by
 *     out[i]   = max(A + s0, B + s1, w[i])
 *     out[i-1] = s0.
 * This is an affine map in maximum-and-addition arithmetic: s' = M (x) s (+) c, where the
 * matrix product (M (x) s)_k = max_j (M[k][j] + s[j]) uses maximum in place of the sum and
 * addition in place of the product, and (+) combines by elementwise maximum. For step i,
 *     M = [[A,   B  ],      c = (w[i],
 *          [0,   NEG]]           NEG ).
 * Two such maps compose into one of the same form (apply g, then f):
 *     (f o g).M = f.M (x) g.M,   (f o g).c = (f.M (x) g.c) (+) f.c,
 * and this composition is associative, so the classic two-pass block scan works with the
 * map as the block summary. It is not the ordinary additive scan: the product here uses
 * maximum and addition, not addition and multiplication. */

typedef struct { double M[2][2]; double c[2]; } Map;

/* s' = M (x) s (+) c  -- apply the affine map to a state. */
static inline void apply_map(const Map *f, const double *s, double *out)
{
    double r0 = f->M[0][0] + s[0];
    double r0b = f->M[0][1] + s[1];
    if (r0b > r0) r0 = r0b;
    if (f->c[0] > r0) r0 = f->c[0];
    double r1 = f->M[1][0] + s[0];
    double r1b = f->M[1][1] + s[1];
    if (r1b > r1) r1 = r1b;
    if (f->c[1] > r1) r1 = f->c[1];
    out[0] = r0;
    out[1] = r1;
}

/* h = f o g  (do g first, then f), same affine form. */
static inline Map compose(const Map *f, const Map *g)
{
    Map h;
    for (int i = 0; i < 2; i++)
        for (int k = 0; k < 2; k++) {
            double a = f->M[i][0] + g->M[0][k];
            double b = f->M[i][1] + g->M[1][k];
            h.M[i][k] = a > b ? a : b;
        }
    for (int i = 0; i < 2; i++) {
        double a = f->M[i][0] + g->c[0];
        double b = f->M[i][1] + g->c[1];
        double m = a > b ? a : b;
        h.c[i] = m > f->c[i] ? m : f->c[i];
    }
    return h;
}

/* The affine map for one step with stimulus wi. */
static inline Map step_map(double wi)
{
    Map f = {{{DECAY_A, DECAY_B}, {0.0, DECAY_NEG}}, {wi, DECAY_NEG}};
    return f;
}

void decay(const double *w, double *out, long n)
{
    if (n <= 0)
        return;
    Map *block = NULL;      /* one composed map per thread */
    double *enter = NULL;   /* the (out[i-1], out[i-2]) pair entering each thread's block */
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        #pragma omp single
        {
            block = malloc(sizeof(Map) * (size_t)nt);
            enter = malloc(sizeof(double) * 2 * (size_t)nt);
        }
        /* the single's implicit barrier makes block and enter visible to every thread */

        const long lo = n * tid / nt;
        const long hi = n * (tid + 1) / nt;

        /* pass 1: fold this thread's block into one affine map */
        Map acc = {{{0.0, DECAY_NEG}, {DECAY_NEG, 0.0}}, {DECAY_NEG, DECAY_NEG}}; /* identity */
        for (long i = lo; i < hi; i++) {
            Map f = step_map(w[i]);
            acc = compose(&f, &acc);
        }
        block[tid] = acc;
        #pragma omp barrier

        /* apply the thread maps in order to the initial state to get each block's entry */
        #pragma omp single
        {
            double s[2] = {DECAY_NEG, DECAY_NEG};   /* out[-1], out[-2] */
            for (int t = 0; t < nt; t++) {
                enter[2 * t] = s[0];
                enter[2 * t + 1] = s[1];
                double ns[2];
                apply_map(&block[t], s, ns);
                s[0] = ns[0];
                s[1] = ns[1];
            }
        }
        /* the single's implicit barrier orders the entry states before pass 2 */

        /* pass 2: run the plain recurrence over this block from its entry state */
        double s0 = enter[2 * tid];       /* out[lo-1] */
        double s1 = enter[2 * tid + 1];   /* out[lo-2] */
        for (long i = lo; i < hi; i++) {
            double x = DECAY_A + s0;
            double xb = DECAY_B + s1;
            if (xb > x) x = xb;
            if (w[i] > x) x = w[i];
            out[i] = x;
            s1 = s0;
            s0 = x;
        }
    }
    free(block);
    free(enter);
}
