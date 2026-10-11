#include "kernel_c.h"
#include <omp.h>
#include <stdlib.h>

void kernel_c(const double * restrict in, double * restrict out, long n)
{
    if (n <= 0) return;
    
    double *sums;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        
        #pragma omp single
        {
            sums = malloc(nt * sizeof(double));
        }
        
        long chunk = (n + nt - 1) / nt;
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        
        // Local scan with aggressive unrolling
        double s = 0.0;
        long i;
        for (i = start; i + 7 < end; i += 8) {
            double v0 = in[i];
            double v1 = in[i + 1];
            double v2 = in[i + 2];
            double v3 = in[i + 3];
            double v4 = in[i + 4];
            double v5 = in[i + 5];
            double v6 = in[i + 6];
            double v7 = in[i + 7];
            
            s += v0; out[i] = s;
            s += v1; out[i + 1] = s;
            s += v2; out[i + 2] = s;
            s += v3; out[i + 3] = s;
            s += v4; out[i + 4] = s;
            s += v5; out[i + 5] = s;
            s += v6; out[i + 6] = s;
            s += v7; out[i + 7] = s;
        }
        for (; i < end; i++) {
            s += in[i];
            out[i] = s;
        }
        sums[tid] = s;
        
        #pragma omp barrier
        
        // Scan sums
        #pragma omp single
        {
            for (int j = 1; j < nt; j++) {
                sums[j] += sums[j - 1];
            }
        }
        
        // Add offset with aggressive unrolling
        if (tid > 0) {
            double off = sums[tid - 1];
            for (i = start; i + 7 < end; i += 8) {
                out[i] += off;
                out[i + 1] += off;
                out[i + 2] += off;
                out[i + 3] += off;
                out[i + 4] += off;
                out[i + 5] += off;
                out[i + 6] += off;
                out[i + 7] += off;
            }
            for (; i < end; i++) {
                out[i] += off;
            }
        }
    }
    
    free(sums);
}
