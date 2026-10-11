// test_kernel_a.c — unit test for the parallel kernel_a implementation.

#include <stdio.h>
#include <math.h>
#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(void) {
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result matches reference");
    return TEST_SUMMARY();
}
