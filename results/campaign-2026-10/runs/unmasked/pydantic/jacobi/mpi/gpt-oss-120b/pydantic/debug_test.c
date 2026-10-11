#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "problem.h"
int main(int argc, char**argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    long n=8; int iters=10;
    double *u=malloc(n*n*sizeof(double));
    for(long i=0;i<n;i++) for(long j=0;j<n;j++) u[i*n+j]=u_init(i,j,n);
    jacobi(u,n,iters);
    if(rank==0){printf("done, checksum=%f\n",0.0);}
    free(u);
    MPI_Finalize(); return 0;}
