// integrate.cu: CUDA kernel for midpoint-rule integration of 4/(1+x^2) over [a, b]
__global__ void integrate_kernel(double *partial_sums, double a, double b, long n) {
    // Each thread accumulates its share of the subintervals
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; // Only process subintervals within bounds
    double h = (b - a) / n;
    double x = a + (idx + 0.5) * h;
    partial_sums[idx] = 4.0 / (1.0 + x * x) * h;
}

// Host function to launch the kernel and combine the partial sums
extern "C" double integrate(double a, double b, long n) {
    int num_blocks = (n + 255) / 256; // 256 threads per block
    int num_threads = 256;
    size_t size = n * sizeof(double);
    double *partial_sums;
    cudaMalloc((void **)&partial_sums, size);
    integrate_kernel<<<num_blocks, num_threads>>>(partial_sums, a, b, n);
    cudaDeviceSynchronize();
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += partial_sums[i];
    }
    cudaFree(partial_sums);
    return sum;
}
