/* The serial routine of matmul.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 17, 100, 300, 1024]. */
#include "matmul.h"

void matmul(const double *A, const double *B, double *C, long n) {
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
