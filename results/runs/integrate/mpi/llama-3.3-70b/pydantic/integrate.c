#include "integrate.h"
#include <mpi.h>
#include <math.h>
double integrate(double a, double b, long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    double h = (b - a) / n;
    double s = 0.0;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }
    double result;
    MPI_Allreduce(&s, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result * h / size;
}
