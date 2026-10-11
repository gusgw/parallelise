#ifndef HISTOGRAM_H
#define HISTOGRAM_H

/* Histogram of n values x[0..n-1], all in [0, 1), into nbins equal-width bins. The
 * kernel first zeroes counts[0..nbins-1], then for each value increments
 * counts[(int)(x[i] * nbins)], clamping the bin index to nbins - 1.
 *
 * Implement THIS EXACT signature in histogram.c. It is a COLLECTIVE routine: every MPI
 * rank calls it together with identical x (each rank holds the full array), and on
 * return every rank holds the full counts. Assume MPI is already initialised by the
 * caller (main); do NOT call MPI_Init or MPI_Finalize inside it. Decompose the value
 * indices across the ranks (each rank bins only its block [lo, hi) into its own counts
 * array), then combine the per-rank counts into the full histogram on every rank with
 * an element-wise sum reduction (MPI_Allreduce of nbins MPI_LONG values with MPI_SUM). */
void histogram(const double *x, long n, long *counts, int nbins);

#endif /* HISTOGRAM_H */
