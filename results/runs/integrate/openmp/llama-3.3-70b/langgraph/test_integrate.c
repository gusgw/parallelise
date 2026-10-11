#include <math.h>
#include "integrate.h"
#include "reference.h"

int main() {
    double result;
    integrate(0.0, 1.0, N_TEST, &result);
    if (fabs(result - EXPECTED) > REF_TOL) {
        return 1;
    }
    return 0;
}
