#include "cg.h"
#include "reference.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int main() {
    double *x = (double *)malloc(REF_M * REF_M * sizeof(double));
    double *b = (double *)malloc(REF_M * REF_M * sizeof(double));
    long i;
    
    srand(time(NULL));
    for (i = 0; i < REF_M * REF_M; i++) {
        b[i] = (double)rand() / RAND_MAX;
    }
    
    cgsolve(x, b, REF_M, REF_ITERS);
    
    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (i = 0; i < REF_M * REF_M; i++) {
        checksum += x[i];
    }
    
    if (fabs(checksum - REF_CHECKSUM) > REF_TOL * 2) {
        printf("Checksum mismatch: expected %f, got %f\n", REF_CHECKSUM, checksum);
        return 1;
    }
    
    printf("Test passed\n");
    free(x);
    free(b);
    return 0;
}
