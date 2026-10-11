#include <stdio.h>
#include <stdlib.h>

#include "kernel_a.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    double result = kernel_a(REF_A, REF_B, REF_N_TEST);
    CHECK_CLOSE(result, REF_EXPECTED, REF_TOL, "kernel_a result matches reference");
    return TEST_SUMMARY();
}
