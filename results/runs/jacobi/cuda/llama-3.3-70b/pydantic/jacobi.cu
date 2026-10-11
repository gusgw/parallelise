// implement the Jacobi 5-point heat stencil in CUDA
__global__ void jacobi_kernel(double *u, double *un, long n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    if (i == 0 || j == 0 || i == n - 1 || j == n - 1) {
        un[i * n + j] = u[i * n + j];
    } else {
        un[i * n + j] = 0.25 * (u[(i-1) * n + j] + u[(i+1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
    }
}

extern "C" void jacobi(double *u, long n, int iters) {
    double *d_u, *d_un;
    cudaMalloc((void **)&d_u, n * n * sizeof(double));
    cudaMalloc((void **)&d_un, n * n * sizeof(double));
    cudaMemcpy(d_u, u, n * n * sizeof(double), cudaMemcpyHostToDevice);
    for (int iter = 0; iter < iters; iter++) {
        jacobi_kernel<<<dim3((n+15)/16, (n+15)/16), dim3(16, 16)>>>(d_u, d_un, n);
        cudaDeviceSynchronize();
        double *tmp = d_u;
        d_u = d_un;
        d_un = tmp;
    }
    cudaMemcpy(u, d_u, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    cudaFree(d_un);
}
