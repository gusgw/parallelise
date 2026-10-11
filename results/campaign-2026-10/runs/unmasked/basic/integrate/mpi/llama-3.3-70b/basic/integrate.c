#include "integrate.h"
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
double integrate(double a, double b, long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    long i;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long start = n * rank / size;
    long end = n * (rank + 1) / size;
    for (i = start; i < end; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    double global_sum;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum * h;
}
