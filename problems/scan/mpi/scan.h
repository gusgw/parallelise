#ifndef SCAN_H
#define SCAN_H

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Implement THIS EXACT signature in scan.c. It is a COLLECTIVE routine: every MPI
 * rank calls it together with identical in (each rank holds a full copy). The result
 * stays DISTRIBUTED: on return, rank r must hold correct values for exactly its own
 * contiguous block out[lo, hi), lo = n*rank/size, hi = n*(rank+1)/size; entries
 * outside the rank's own block are unspecified. Do NOT gather the whole array inside
 * this function -- moving the full result to every rank costs as much as computing
 * it, so a caller who needs it gathered does that separately. Assume MPI is already
 * initialised by the caller (main); do NOT call MPI_Init or MPI_Finalize inside it.
 *
 * The naive loop is inherently sequential (each out[i] depends on out[i-1]), so use
 * the classic block-scan decomposition: each rank scans its own block of in locally;
 * an MPI_Exscan (MPI_SUM) of the block totals gives each rank's starting offset
 * (rank 0's offset is 0); add the offset to the local block. */
void scan(const double *in, double *out, long n);

#endif /* SCAN_H */
