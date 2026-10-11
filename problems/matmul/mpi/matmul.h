#ifndef MATMUL_H
#define MATMUL_H

/* Dense matrix multiply C = A * B for n x n row-major matrices.
 *
 * Implement THIS EXACT signature in matmul.c. It is a COLLECTIVE routine: every MPI
 * rank calls it together with identical A and B (each rank holds full copies), and on
 * return every rank holds the full result C. Assume MPI is already initialised by the
 * caller (main); do NOT call MPI_Init or MPI_Finalize inside it. Decompose the rows of
 * C across the ranks (each rank computes only its block of rows) and assemble the full
 * C on every rank with a gather (MPI_Allgather / MPI_Allgatherv). */
void matmul(const double *A, const double *B, double *C, long n);

#endif /* MATMUL_H */
