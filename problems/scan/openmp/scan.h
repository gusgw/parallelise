#ifndef SCAN_H
#define SCAN_H

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Implement THIS EXACT signature in scan.c. The naive loop is inherently sequential
 * (each out[i] depends on out[i-1]), so parallelise with the classic two-pass block
 * scan: pass 1, each thread sums its contiguous block of in; a serial exclusive scan
 * of the block sums gives each block's starting offset; pass 2, each thread scans its
 * own block, starting from its offset. Plain shared-memory function: no MPI. */
void scan(const double *in, double *out, long n);

#endif /* SCAN_H */
