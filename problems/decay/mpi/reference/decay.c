#include <mpi.h>
#include <stdlib.h>

#include "decay.h"
#include "problem.h"   /* DECAY_A, DECAY_B, DECAY_NEG */

/* Reference solution (validates the harness -- NOT shown to the agents).
 *
 * The recurrence out[i] = max(A + out[i-1], B + out[i-2], w[i]) carries the pair of
 * most-recent scores. In maximum-and-addition arithmetic one step is the affine map
 *     s' = M (x) s (+) c,   (M (x) s)_k = max_j (M[k][j] + s[j]),   with
 *     M = [[A, B], [0, NEG]],   c = (w[i], NEG),
 * and two maps compose into one of the same form. So each rank folds its own block into
 * one such map, the ranks exchange their maps (there are only `size` of them), and each
 * rank composes the maps of all lower ranks -- applied to the initial pair (NEG, NEG) --
 * to get the pair entering its block. It then runs the plain recurrence over its block.
 * The result stays distributed per the contract. This is not the ordinary additive scan:
 * the product here uses maximum and addition, not addition and multiplication. */

typedef struct { double M[2][2]; double c[2]; } Map;   /* 6 contiguous doubles */

static inline void apply_map(const Map *f, const double *s, double *out)
{
    double r0 = f->M[0][0] + s[0], r0b = f->M[0][1] + s[1];
    if (r0b > r0) r0 = r0b;
    if (f->c[0] > r0) r0 = f->c[0];
    double r1 = f->M[1][0] + s[0], r1b = f->M[1][1] + s[1];
    if (r1b > r1) r1 = r1b;
    if (f->c[1] > r1) r1 = f->c[1];
    out[0] = r0;
    out[1] = r1;
}

/* h = f o g (do g first, then f). */
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

void decay(const double *w, double *out, long n)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long lo = n * rank / size;
    const long hi = n * (rank + 1) / size;

    /* fold this rank's block into one affine map (identity if the block is empty) */
    Map acc = {{{0.0, DECAY_NEG}, {DECAY_NEG, 0.0}}, {DECAY_NEG, DECAY_NEG}};
    for (long i = lo; i < hi; i++) {
        Map f = {{{DECAY_A, DECAY_B}, {0.0, DECAY_NEG}}, {w[i], DECAY_NEG}};
        acc = compose(&f, &acc);
    }

    /* exchange the per-rank maps (6 doubles each) so every rank has all of them */
    Map *all = malloc(sizeof(Map) * (size_t)size);
    MPI_Allgather(&acc, 6, MPI_DOUBLE, all, 6, MPI_DOUBLE, MPI_COMM_WORLD);

    /* compose the maps of all lower ranks, applied to the initial pair, to get the pair
     * entering this rank's block */
    double s[2] = {DECAY_NEG, DECAY_NEG};   /* out[-1], out[-2] */
    for (int t = 0; t < rank; t++) {
        double ns[2];
        apply_map(&all[t], s, ns);
        s[0] = ns[0];
        s[1] = ns[1];
    }
    free(all);

    /* run the plain recurrence over this rank's block from its entry pair */
    double s0 = s[0], s1 = s[1];
    for (long i = lo; i < hi; i++) {
        double x = DECAY_A + s0, xb = DECAY_B + s1;
        if (xb > x) x = xb;
        if (w[i] > x) x = w[i];
        out[i] = x;
        s1 = s0;
        s0 = x;
    }
    /* the result stays distributed: this rank's block of out is correct -- no gather */
}
