/* test_nbody.c - Unit test for nbody CUDA kernel */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *px = malloc(sizeof(double) * n);
    double *py = malloc(sizeof(double) * n);
    double *ax = malloc(sizeof(double) * n);
    double *ay = malloc(sizeof(double) * n);
    
    if (!px || !py || !ax || !ay) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    
    /* Fill positions deterministically */
    for (long i = 0; i < n; i++) {
        px[i] = px_of(i);
        py[i] = py_of(i);
    }
    
    /* Compute accelerations on GPU */
    nbody_accel(px, py, ax, ay, n);
    
    /* Compute checksum */
    double cs = checksum(ax, ay, n);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch with reference");
    
    free(px);
    free(py);
    free(ax);
    free(ay);
    
    return TEST_SUMMARY();
}
