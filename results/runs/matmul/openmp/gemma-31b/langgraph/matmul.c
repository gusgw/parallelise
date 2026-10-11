#include "matmul.h"
#include <omp.h>

void matmul(const double *A, const double *B, double *C, long n) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            double sum = 0.0;
            for (long k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
