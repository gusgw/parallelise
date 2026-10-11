#include "kernel_d.h"
#include "problem.h"
#include <omp.h>
#include <stdlib.h>

void cgsolve(double *x, const double *b, long m, int iters)
{
    long t;
    double rsold = 0.0, rsnew = 0.0;
    double *rbuf = (double *)malloc(m * m * sizeof(double));
    double *pbuf = (double *)malloc(m * m * sizeof(double));
    double *abuf = (double *)malloc(m * m * sizeof(double));

    #pragma omp parallel for
    for (t = 0; t < m * m; t++) rbuf[t] = b[t];

    for (int iter = 0; iter < iters; iter++) {
        double ap = 0.0;
        #pragma omp parallel for reduction(+:rsnew)
        for (t = 0; t < m * m; t++) {
            pbuf[t] = rbuf[t];
            rsnew += rbuf[t] * rbuf[t];
        }

        if (iter > 0) rsnew /= rsold;
        else rsnew = 1.0;

        #pragma omp parallel for
        for (t = 0; t < m * m; t++) pbuf[t] = rbuf[t] + rsnew * pbuf[t];

        #pragma omp parallel for
        for (t = 0; t < m * m; t++) abuf[t] = pbuf[t];

        #pragma omp parallel for reduction(+:ap)
        for (t = 0; t < m * m; t++) {
            abuf[t] = (abuf[t] * 2.0) - b[t];
            ap += abuf[t] * pbuf[t];
        }

        #pragma omp parallel for
        for (t = 0; t < m * m; t++) rbuf[t] = rbuf[t] - (ap / (rsold + 1.0e-10)) * pbuf[t];

        rsold = rsnew;
    }

    #pragma omp parallel for
    for (t = 0; t < m * m; t++) x[t] = rbuf[t];

    free(rbuf);
    free(pbuf);
    free(abuf);
}
