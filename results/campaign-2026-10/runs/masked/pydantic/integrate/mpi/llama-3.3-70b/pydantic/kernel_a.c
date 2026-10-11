#include "kernel_a.h"
#include "mpi.h"

double kernel_a(double a, double b, long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    for (long i = lo; i < hi; i++) {
        double x = a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    double partial_sum;
    MPI_Allreduce(&sum, &partial_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return h * partial_sum;
}
