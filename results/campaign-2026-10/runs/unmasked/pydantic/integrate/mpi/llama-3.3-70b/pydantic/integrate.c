#include "integrate.h"
#include "reference.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double integrate(double a, double b, long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    long i;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    for (i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    double global_sum;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum * h;
}
