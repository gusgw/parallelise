#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "kernel_d.h"
#include "problem.h"
int main(int argc, char**argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    long m=64; long n=m*m; double *b=malloc(n*sizeof(double)); double *x=malloc(n*sizeof(double));
    for(long i=0;i<n;i++) b[i]=b_of(i);
    cgsolve(x,b,m,1);
    if(rank==0) printf("done 1 iter\n");
    MPI_Finalize();
    return 0;
}
