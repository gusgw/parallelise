#ifndef HISTOGRAM_H
#define HISTOGRAM_H

/* Histogram of n values x[0..n-1], all in [0, 1), into nbins equal-width bins. The
 * kernel first zeroes counts[0..nbins-1], then for each value increments
 * counts[(int)(x[i] * nbins)], clamping the bin index to nbins - 1.
 *
 * Implement THIS EXACT signature in histogram.c. Parallelise with OpenMP. Beware the
 * data race: two threads must never increment the same shared bin at once (a race
 * loses increments and gives silently wrong counts). Give each thread a PRIVATE local
 * histogram for the hot loop and merge the local histograms into the shared counts at
 * the end, inside a critical section or with atomics on the merge only. Plain
 * shared-memory function: no MPI. */
void histogram(const double *x, long n, long *counts, int nbins);

#endif /* HISTOGRAM_H */
