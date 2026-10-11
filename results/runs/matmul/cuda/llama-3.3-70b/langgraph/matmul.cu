#include "matmul.h"
#include "reference.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < N && idy < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[idy * N + k] * B[k * N + idx];
        }
        C[idy * N + idx] = sum;
    }
}

float matmul(float *A, float *B, float *C, int N) {
    int numBlocksX = (N + 15) / 16;
    int numBlocksY = (N + 15) / 16;
    dim3 dimBlock(16, 16);
    dim3 dimGrid(numBlocksX, numBlocksY);
    matmul_kernel<<<dimGrid, dimBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    return 0;
}