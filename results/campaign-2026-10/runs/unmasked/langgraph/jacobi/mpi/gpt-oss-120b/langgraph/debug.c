#include <mpi.h>
#include <stdio.h>
#include "jacobi.h"
int main(int argc, char**argv){
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    long n = 1024;
    long lo = n * rank / size;
    long hi = n * (rank + 1) / size;
    long local_n = hi - lo;
    printf("rank %d local_n %ld\n", rank, local_n);
    MPI_Finalize();
    return 0;
}
