/* The serial routine of decay.ipynb, translated from its Python by the model kimi-k2.5 at temperature 0 and accepted on 2026-11-06 at the sizes [1, 1000, 100000, 60000000]. */
#include "decay.h"

#define NO_SCORE -1.0e9

void decay(const double *w, double *out, long n) {
    double last = NO_SCORE;
    double before_last = NO_SCORE;
    
    for (long i = 0; i < n; i++) {
        double a = last - 2.0;
        double b = before_last - 1.0;
        double c = w[i];
        
        // Compute max of three values
        double max_val = a;
        if (b > max_val) {
            max_val = b;
        }
        if (c > max_val) {
            max_val = c;
        }
        
        out[i] = max_val;
        before_last = last;
        last = max_val;
    }
}
