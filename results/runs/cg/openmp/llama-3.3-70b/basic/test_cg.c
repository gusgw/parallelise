#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    int m = REF_M;
    int iters = REF_ITERS;
    int n = m * m;
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    
    // Initialize b
    for (int i = 0; i < n; i++) {
        b[i] = ((i * 97) % 211) / 211.0;
    }
    
    cgsolve(b, x, m, iters);
    
    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += x[i];
    }
    
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL) {
        printf("Error: checksum mismatch\n");
        return 1;
    }
    
    printf("Checksum match\n");
    return 0;
}
