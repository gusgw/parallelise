#ifndef MATMUL_H
#define MATMUL_H

/* Dense matrix multiply C = A * B for n x n row-major matrices.
 *
 * Implement THIS EXACT signature in matmul.c. Parallelise with OpenMP (e.g. a
 * '#pragma omp parallel for' over the rows of C; each row of C is independent).
 * Plain shared-memory function: no MPI. */
void matmul(const double *A, const double *B, double *C, long n);

#endif /* MATMUL_H */
