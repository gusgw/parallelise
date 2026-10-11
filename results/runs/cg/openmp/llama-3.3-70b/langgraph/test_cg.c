#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    int m = REF_M;
    int iters = REF_ITERS;
    int n = m * m;
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double checksum;
    int i;
    
    // fill b
    for (i = 0; i < n; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }
    
    cg_solve(x, b, m, iters);
    
    checksum = 0.0;
    for (i = 0; i < n; i++) {
        checksum += x[i];
    }
    
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("ERROR: checksum mismatch\n");
        return 1;
    }
    
    printf("OK\n");
    return 0;
}
